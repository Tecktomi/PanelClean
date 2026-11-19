"""Generador de pares sintéticos para evaluar PanelClean (tarea T1.4.1).

Cada par es un fondo limpio con texto japonés vertical dibujado encima. Como se
conoce el fondo, se sabe exactamente qué píxeles cambió el texto (la máscara de
referencia) y qué había debajo de ellos. Eso permite medir la máscara (IoU/Dice)
y la limpieza (MAE/PSNR/SSIM), cosa imposible con manga real.

Los fondos se generan por procedimiento, así que su clase se conoce por
construcción:
- Uniforme: blanco o gris liso con ruido de escaneo.
- Estructurado: trama de puntos, rayado paralelo o degradado.
- Complejo: trazos irregulares, manchas negras y parches de trama (imita dibujo).

Las frases salen de las anotaciones de OpenMantra (datasets/open-mantra) y la
fuente es Noto Sans JP Bold (evaluacion/fuentes, licencia SIL OFL 1.1).

Para mirar algunos pares (desde la raíz del repositorio):
    python -m evaluacion.sinteticos --por-clase 3 --salida pares_ejemplo
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CARPETA = Path(__file__).resolve().parent
FUENTE = CARPETA / 'fuentes' / 'NotoSansJP-Bold.otf'
ANOTACIONES = CARPETA.parent / 'datasets' / 'open-mantra' / 'annotation.json'

CLASES = ('Uniforme', 'Estructurado', 'Complejo')
ALTO, ANCHO = 300, 360
# Diferencia mínima (en niveles de gris, de 0 a 255) para que un píxel cuente
# como «cambiado por el texto» en la máscara de referencia.
UMBRAL_REFERENCIA = 8
# Margen de la selección alrededor del texto, como un rectángulo ajustado del usuario.
MARGEN_SELECCION = 10
SEMILLA = 2026

# Caracteres que en escritura vertical se giran 90° (alargamiento, puntos
# suspensivos, paréntesis y comillas).
GIRAR = set('ー〜～…‥―－-—「」『』（）()【】〈〉《》［］[]')
# Puntuación que va arriba a la derecha de su casilla.
PUNTUACION = set('、。，．')
# Kana pequeños: van un poco hacia arriba a la derecha.
PEQUENOS = set('ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ')


@dataclass
class Par:
    """Un par sintético: la entrada de PanelClean y todo lo que se sabe de ella."""

    limpio: np.ndarray  # fondo sin texto (BGR)
    con_texto: (
        np.ndarray
    )  # mismo fondo con el texto encima (BGR): entrada de PanelClean
    referencia: np.ndarray  # 255 donde el texto cambió el fondo (máscara exacta)
    caja: tuple[int, int, int, int]  # selección (x1, y1, x2, y2) alrededor del texto
    clase: str  # clase del fondo, conocida por construcción
    texto: str
    borde_blanco: bool  # el texto lleva contorno blanco (habitual sobre dibujo)
    semilla: int


# ---------------------------------------------------------------------------
# Frases y fuente
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def frases() -> tuple[str, ...]:
    """Frases en japonés de OpenMantra, normalizadas para escribirlas en vertical."""
    datos = json.loads(ANOTACIONES.read_text(encoding='utf-8'))
    salida = []
    for libro in datos:
        for pagina in libro['pages']:
            for caja in pagina['text']:
                texto = caja['text_ja'].replace('...', '…').replace('..', '‥')
                texto = texto.replace('!', '！').replace('?', '？')
                texto = ''.join(c for c in texto if not c.isspace() and c != '.')
                if len(texto) >= 3:
                    salida.append(texto)
    return tuple(salida)


@functools.lru_cache(maxsize=16)
def fuente(tam: int) -> ImageFont.FreeTypeFont:
    """Noto Sans JP Bold en el tamaño pedido (en píxeles)."""
    return ImageFont.truetype(str(FUENTE), tam)


def _casilla(caracter: str, tam: int, borde: int) -> tuple[np.ndarray, np.ndarray]:
    """Dibuja un carácter en una casilla cuadrada.

    Devuelve (relleno, contorno) como alfas en [0, 1]. El contorno incluye el
    relleno ensanchado borde píxeles (0 = sin contorno).
    """
    lado = tam + 2 * (borde + 3)
    centro = (lado / 2, lado / 2)
    capas = []
    for ancho in (0, borde):
        lienzo = Image.new('L', (lado, lado), 0)
        ImageDraw.Draw(lienzo).text(
            centro,
            caracter,
            font=fuente(tam),
            fill=255,
            anchor='mm',
            stroke_width=ancho,
            stroke_fill=255,
        )
        if caracter in GIRAR:
            lienzo = lienzo.rotate(-90)
        capas.append(np.asarray(lienzo, np.float32) / 255)
    return capas[0], capas[1]


def texto_vertical(
    texto: str, tam: int, por_columna: int, borde: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compone el texto en columnas verticales que se leen de derecha a izquierda.

    Devuelve (relleno, contorno) como alfas en [0, 1] del tamaño del bloque.
    """
    columnas = [texto[i : i + por_columna] for i in range(0, len(texto), por_columna)]
    paso_x = int(round(tam * 1.35))
    paso_y = int(round(tam * 1.05))
    lado = tam + 2 * (borde + 3)
    alto = (por_columna - 1) * paso_y + lado
    ancho = (len(columnas) - 1) * paso_x + lado
    relleno = np.zeros((alto, ancho), np.float32)
    contorno = np.zeros((alto, ancho), np.float32)
    for c, columna in enumerate(columnas):
        x = (len(columnas) - 1 - c) * paso_x  # la primera columna va a la derecha
        for f, caracter in enumerate(columna):
            y = f * paso_y
            dx = dy = 0
            if caracter in PUNTUACION:
                dx, dy = int(tam * 0.3), -int(tam * 0.3)
            elif caracter in PEQUENOS:
                dx, dy = int(tam * 0.1), -int(tam * 0.1)
            r, k = _casilla(caracter, tam, borde)
            _pegar(relleno, r, x + dx, y + dy)
            _pegar(contorno, k, x + dx, y + dy)
    return relleno, contorno


def _pegar(destino: np.ndarray, capa: np.ndarray, x: int, y: int) -> None:
    """Combina capa sobre destino en (x, y) con el máximo, recortando a los bordes."""
    h, w = capa.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(destino.shape[1], x + w), min(destino.shape[0], y + h)
    if x1 < x2 and y1 < y2:
        zona = destino[y1:y2, x1:x2]
        np.maximum(zona, capa[y1 - y : y2 - y, x1 - x : x2 - x], out=zona)


# ---------------------------------------------------------------------------
# Fondos
# ---------------------------------------------------------------------------


def _jpeg(gris: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pasa el fondo por JPEG (como una página escaneada) y lo devuelve en BGR."""
    u8 = np.clip(np.round(gris), 0, 255).astype(np.uint8)
    calidad = int(rng.integers(80, 96))
    ok, datos = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    assert ok
    return cv2.cvtColor(cv2.imdecode(datos, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)


def _coordenadas(angulo_grados: float) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas (u, v) de cada píxel, giradas en el ángulo dado."""
    yy, xx = np.mgrid[0:ALTO, 0:ANCHO].astype(np.float32)
    a = math.radians(angulo_grados)
    return xx * math.cos(a) + yy * math.sin(a), -xx * math.sin(a) + yy * math.cos(a)


def _patron_puntos(rng: np.random.Generator) -> np.ndarray:
    """Trama de puntos (screentone): 1 donde hay punto, con bordes suaves."""
    periodo = rng.uniform(4.5, 8.0)
    u, v = _coordenadas(float(rng.choice([0.0, 45.0])) + rng.uniform(-4, 4))
    s = (np.cos(2 * np.pi * u / periodo) + np.cos(2 * np.pi * v / periodo)) / 2
    corte = rng.uniform(-0.1, 0.6)
    return np.clip((s - corte) / 0.25, 0, 1)


def _patron_rayas(rng: np.random.Generator) -> np.ndarray:
    """Rayado paralelo en un ángulo al azar: 1 sobre la raya, con bordes suaves."""
    periodo = rng.uniform(5.0, 10.0)
    u, _ = _coordenadas(rng.uniform(0, 180))
    s = np.cos(2 * np.pi * u / periodo)
    corte = rng.uniform(0.2, 0.7)
    return np.clip((s - corte) / 0.25, 0, 1)


def fondo_uniforme(rng: np.random.Generator) -> np.ndarray:
    """Blanco o gris claro liso con un poco de ruido."""
    base = rng.uniform(228, 252) if rng.random() < 0.75 else rng.uniform(195, 228)
    gris = np.full((ALTO, ANCHO), base, np.float32)
    gris += rng.normal(0, rng.uniform(0.5, 2.5), gris.shape)
    return _jpeg(gris, rng)


def fondo_estructurado(rng: np.random.Generator) -> np.ndarray:
    """Trama de puntos, rayado paralelo o degradado suave."""
    blanco = rng.uniform(232, 252)
    tipo = rng.choice(['puntos', 'rayas', 'degradado'], p=[0.4, 0.4, 0.2])
    if tipo == 'degradado':
        u, _ = _coordenadas(rng.uniform(0, 360))
        t = (u - u.min()) / max(1e-6, float(u.max() - u.min()))
        oscuro = rng.uniform(110, 190)
        gris = blanco + (oscuro - blanco) * t
    else:
        patron = _patron_puntos(rng) if tipo == 'puntos' else _patron_rayas(rng)
        tono = rng.uniform(30, 160)
        gris = blanco - (blanco - tono) * patron
    gris += rng.normal(0, 1.5, gris.shape)
    return _jpeg(gris.astype(np.float32), rng)


def _curva(rng: np.random.Generator) -> np.ndarray:
    """Puntos de una curva de Bézier cúbica al azar que cruza la imagen."""
    p = rng.uniform([-40, -40], [ANCHO + 40, ALTO + 40], size=(4, 2))
    t = np.linspace(0, 1, 60)[:, None]
    puntos = (
        (1 - t) ** 3 * p[0]
        + 3 * (1 - t) ** 2 * t * p[1]
        + 3 * (1 - t) * t**2 * p[2]
        + t**3 * p[3]
    )
    return np.round(puntos).astype(np.int32)


def fondo_complejo(rng: np.random.Generator) -> np.ndarray:
    """Imita dibujo: parches de trama, manchas oscuras y muchos trazos curvos."""
    gris = np.full((ALTO, ANCHO), rng.uniform(235, 252), np.float32)
    # Parches de trama con forma irregular.
    for _ in range(int(rng.integers(1, 4))):
        zona = np.zeros((ALTO, ANCHO), np.uint8)
        cv2.fillPoly(zona, [_curva(rng)], 255)
        patron = _patron_puntos(rng) if rng.random() < 0.6 else _patron_rayas(rng)
        tono = rng.uniform(40, 150)
        gris = np.where(zona > 0, gris - (gris - tono) * patron, gris)
    # Manchas oscuras (pelo, sombras).
    for _ in range(int(rng.integers(0, 3))):
        zona = np.zeros((ALTO, ANCHO), np.uint8)
        cv2.fillPoly(zona, [_curva(rng)], 255, lineType=cv2.LINE_AA)
        gris = np.minimum(gris, 255 - zona.astype(np.float32) * rng.uniform(0.75, 1.0))
    # Trazos de dibujo de distinto grosor.
    lienzo = gris.astype(np.uint8)
    for _ in range(int(rng.integers(15, 40))):
        color = int(rng.uniform(0, 70))
        grosor = int(rng.integers(1, 4))
        cv2.polylines(lienzo, [_curva(rng)], False, color, grosor, cv2.LINE_AA)
    return _jpeg(lienzo.astype(np.float32), rng)


FONDOS = {
    'Uniforme': fondo_uniforme,
    'Estructurado': fondo_estructurado,
    'Complejo': fondo_complejo,
}


# ---------------------------------------------------------------------------
# Pares
# ---------------------------------------------------------------------------


def generar_par(semilla: int, clase: str) -> Par:
    """Genera un par con la clase de fondo pedida. La misma semilla da el mismo par."""
    rng = np.random.default_rng(semilla)
    limpio = FONDOS[clase](rng)

    prob_borde = {'Uniforme': 0.0, 'Estructurado': 0.25, 'Complejo': 0.5}[clase]
    borde_blanco = bool(rng.random() < prob_borde)
    grosor_borde = int(rng.integers(2, 4)) if borde_blanco else 0
    tam = int(rng.integers(20, 31))

    # Cuántos caracteres caben por columna y cuántas columnas, con margen.
    lado = tam + 2 * (grosor_borde + 3)
    max_filas = (ALTO - 2 * 30 - lado) // int(round(tam * 1.05)) + 1
    max_cols = min(4, (ANCHO - 2 * 30 - lado) // int(round(tam * 1.35)) + 1)
    por_columna = int(rng.integers(3, max_filas + 1))
    lista = frases()
    texto = lista[int(rng.integers(len(lista)))][: por_columna * max_cols]

    relleno, contorno = texto_vertical(texto, tam, por_columna, grosor_borde)
    h, w = relleno.shape
    x0 = int(rng.integers(20, ANCHO - w - 20 + 1))
    y0 = int(rng.integers(20, ALTO - h - 20 + 1))
    a_relleno = np.zeros((ALTO, ANCHO), np.float32)
    a_contorno = np.zeros((ALTO, ANCHO), np.float32)
    a_relleno[y0 : y0 + h, x0 : x0 + w] = relleno
    a_contorno[y0 : y0 + h, x0 : x0 + w] = contorno

    imagen = limpio.astype(np.float32)
    if borde_blanco:
        imagen = imagen * (1 - a_contorno[..., None]) + 255 * a_contorno[..., None]
    tinta = rng.uniform(0, 20)
    imagen = imagen * (1 - a_relleno[..., None]) + tinta * a_relleno[..., None]
    con_texto = np.clip(np.round(imagen), 0, 255).astype(np.uint8)

    diferencia = np.abs(con_texto.astype(np.int16) - limpio.astype(np.int16)).max(
        axis=2
    )
    referencia = (diferencia > UMBRAL_REFERENCIA).astype(np.uint8) * 255
    ys, xs = np.nonzero(referencia)
    caja = (
        max(0, int(xs.min()) - MARGEN_SELECCION),
        max(0, int(ys.min()) - MARGEN_SELECCION),
        min(ANCHO, int(xs.max()) + 1 + MARGEN_SELECCION),
        min(ALTO, int(ys.max()) + 1 + MARGEN_SELECCION),
    )
    return Par(limpio, con_texto, referencia, caja, clase, texto, borde_blanco, semilla)


def generar_conjunto(por_clase: int = 50, semilla: int = SEMILLA) -> list[Par]:
    """Genera por_clase pares de cada clase de fondo, siempre los mismos para una semilla."""
    pares = []
    for k, clase in enumerate(CLASES):
        for i in range(por_clase):
            pares.append(generar_par(semilla * 10_000 + k * 1_000 + i, clase))
    return pares


def main() -> None:
    """Guarda algunos pares como PNG para mirarlos."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--por-clase', type=int, default=3)
    parser.add_argument('--salida', default='pares_ejemplo')
    args = parser.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    for i, par in enumerate(generar_conjunto(args.por_clase)):
        base = salida / f'{i:03d}_{par.clase.lower()}'
        cv2.imwrite(f'{base}_limpio.png', par.limpio)
        cv2.imwrite(f'{base}_con_texto.png', par.con_texto)
        cv2.imwrite(f'{base}_referencia.png', par.referencia)
    print(f'{i + 1} pares guardados en {salida}')


if __name__ == '__main__':
    main()
