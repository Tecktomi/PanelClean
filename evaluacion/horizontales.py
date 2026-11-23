"""Páginas de OpenMantra con los globos traducidos, en renglones horizontales.

OpenMantra solo trae las páginas en japonés (texto vertical), pero sus anotaciones
incluyen la traducción al inglés de cada texto. Aquí se borran las letras de los
globos blancos y se escribe esa traducción, con fuentes libres del repositorio y
variando el tamaño, el grosor, la inclinación y las mayúsculas. Son datos de
entrenamiento aproximados para el texto latino: ayudan, pero no reemplazan páginas
reales de otros idiomas (para eso está marcar.py).

Solo se usan al medir los candidatos; las imágenes no se guardan.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from panelclean import vision as V

RAIZ = Path(__file__).resolve().parent.parent
FUENTES = (
    RAIZ / 'evaluacion' / 'fuentes' / 'NotoSansJP-Bold.otf',
    RAIZ / 'panelclean' / 'fuentes' / 'Quicksand-Bold.ttf',
)
INTERLINEA = 1.12

Caja = tuple[int, int, int, int]  # (x1, y1, x2, y2)


def azar(libro: str, pagina: int) -> np.random.Generator:
    """Generador propio de cada página: el resultado no depende del orden."""
    return np.random.default_rng([7, pagina, *libro.encode('utf8')])


def _renglones(
    dibujo: ImageDraw.ImageDraw, palabras: list[str], letra, ancho: float
) -> list[str]:
    """Reparte las palabras en renglones que caben en el ancho dado."""
    renglones, actual = [], ''
    for palabra in palabras:
        probado = f'{actual} {palabra}'.strip()
        if not actual or dibujo.textlength(probado, font=letra) <= ancho:
            actual = probado
        else:
            renglones.append(actual)
            actual = palabra
    return renglones + [actual]


def _borrar_letras(
    original: np.ndarray, limpia: np.ndarray, caja: Caja
) -> tuple[np.ndarray, bool]:
    """Borra las letras de una caja; devuelve la imagen y si quedó papel liso."""
    r = V.bounded_region(original, caja)
    V.generate_mask(original, r, V.Options())
    nueva, _, _ = V.clean_region(limpia, r, V.Options(method='Relleno local'))
    zona = cv2.cvtColor(nueva[r.y : r.y + r.h, r.x : r.x + r.w], cv2.COLOR_BGR2GRAY)
    return nueva, float((zona > V.PAPEL_CLARO).mean()) >= 0.9


def traducir(
    imagen: np.ndarray, textos: list[dict], rng: np.random.Generator
) -> tuple[np.ndarray, list[Caja]]:
    """Página con sus globos blancos traducidos y las cajas de texto que le quedan.

    Los textos sobre dibujo (o sin traducción) se dejan en japonés, con su caja
    anotada. La caja de un texto traducido es la de los renglones escritos.
    """
    alto, ancho = imagen.shape[:2]
    limpia, cajas, pendientes = imagen.copy(), [], []
    for t in textos:
        caja = (t['x'], t['y'], t['x'] + t['w'], t['y'] + t['h'])
        frase = t.get('text_en') or ''
        palabras = (frase if rng.random() < 0.25 else frase.upper()).split()
        nueva, es_papel = _borrar_letras(imagen, limpia, caja)
        if palabras and es_papel:
            limpia = nueva
            pendientes.append((t, palabras))
        else:
            cajas.append(caja)
    pagina = Image.fromarray(cv2.cvtColor(limpia, cv2.COLOR_BGR2RGB))
    dibujo = ImageDraw.Draw(pagina)
    ruta = str(FUENTES[int(rng.integers(len(FUENTES)))])
    for t, palabras in pendientes:
        trazo = int(rng.random() < 0.5)
        inclinado = float(rng.uniform(-0.25, 0.25)) if rng.random() < 0.3 else 0.0
        for tam in range(int(rng.integers(20, 27)), 9, -1):
            letra = ImageFont.truetype(ruta, tam)
            renglones = _renglones(dibujo, palabras, letra, 1.05 * t['w'])
            largos = [dibujo.textlength(q, font=letra) for q in renglones]
            if (
                len(renglones) * tam * INTERLINEA <= t['h']
                and max(largos) <= 1.05 * t['w']
            ):
                break
        y = y0 = t['y'] + (t['h'] - len(renglones) * tam * INTERLINEA) / 2
        for renglon, largo in zip(renglones, largos):
            x = t['x'] + (t['w'] - largo) / 2
            capa = Image.new('L', pagina.size, 0)
            ImageDraw.Draw(capa).text(
                (x, y), renglon, font=letra, fill=255, stroke_width=trazo
            )
            if inclinado:
                capa = capa.transform(
                    pagina.size,
                    Image.Transform.AFFINE,
                    (1, inclinado, -inclinado * (y + tam / 2), 0, 1, 0),
                    Image.Resampling.BICUBIC,
                )
            pagina.paste((0, 0, 0), (0, 0), capa)
            y += tam * INTERLINEA
        medio = t['x'] + t['w'] / 2
        cajas.append(
            (
                max(0, int(medio - max(largos) / 2)),
                max(0, int(y0)),
                min(ancho, int(medio + max(largos) / 2) + 1),
                min(alto, int(y) + 1),
            )
        )
    return cv2.cvtColor(np.asarray(pagina), cv2.COLOR_RGB2BGR), cajas
