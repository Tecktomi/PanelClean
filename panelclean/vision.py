"""Pipeline PDI: contornos, segmentación, componentes, morfología y reconstrucción."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


@dataclass
class Options:
    """Ajustes de la máscara y de la limpieza elegidos en la interfaz."""

    threshold: str = 'Otsu'
    sensitivity: int = 4
    # 2 px: cubre el antialias y el ruido de compresión que rodean a las letras.
    # En las cajas de OpenMantra deja la mitad de «fantasmas» que con 1 px.
    expansion: int = 2
    radius: int = 3
    method: str = 'Automático'


@dataclass
class Region:
    """Rectángulo marcado sobre la página, con su máscara y el análisis del fondo.

    interior: máscara opcional del interior del globo (solo en regiones detectadas).
    mask: píxeles a limpiar, del tamaño de la región (255 = limpiar).
    applied_radius: radio de inpainting usado cuando se aplicó la limpieza.
    """

    x: int
    y: int
    w: int
    h: int
    id: int = 0
    numero: int = 0  # número visible en la interfaz (R01, R02…)
    origin: str = 'Manual'
    interior: Optional[np.ndarray] = field(default=None, repr=False)
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    classification: str = 'Sin analizar'
    variance: float = 0.0
    edge_density: float = 0.0
    entropy: float = 0.0
    dark_level: float = 255.0
    texture: float = 0.0
    suggested_method: str = ''
    method_override: str = 'Automático'
    mask_edited: bool = False
    # Probabilidad de que sea texto según el detector (1.0 si la marcó el usuario).
    confianza: float = 1.0
    # Zona de papel (globo) que rodea a una región detectada; ver papel_de_pagina.
    papel: int = 0
    review_reason: str = ''
    last_method: str = ''
    applied_radius: Optional[int] = None

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """Rectángulo como (x, y, ancho, alto)."""
        return self.x, self.y, self.w, self.h

    def contains(self, x: int, y: int) -> bool:
        """Indica si el punto (x, y) de la página está dentro de la región."""
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h


def bgr_from_file(path: str) -> np.ndarray:
    """Lectura de imágenes con rutas que contengan tildes o eñes."""
    from PIL import Image, ImageOps

    with Image.open(path) as obj:
        rgb = np.asarray(ImageOps.exif_transpose(obj).convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_png(path: str, image: np.ndarray) -> None:
    """Guarda la imagen como PNG; funciona con rutas con tildes o eñes en Windows."""
    from pathlib import Path

    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError('No se pudo codificar la imagen')
    Path(path).write_bytes(encoded.tobytes())


def bounded_region(
    image: np.ndarray,
    xyxy: tuple[int, int, int, int],
    id: int = 0,
    origin: str = 'Manual',
    interior: Optional[np.ndarray] = None,
) -> Region:
    """Crea una Region a partir de dos esquinas (x1, y1, x2, y2), en cualquier orden.

    Recorta el rectángulo a los bordes de la imagen y lanza ValueError si queda
    menor a 12 × 12 píxeles.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if x2 - x1 < 12 or y2 - y1 < 12:
        raise ValueError('La región debe tener al menos 12 × 12 píxeles')
    region = Region(x1, y1, x2 - x1, y2 - y1, id=id, origin=origin)
    if interior is not None:
        assert interior.shape == (region.h, region.w)
        region.interior = interior.copy()
    return region


def detect_bubbles(image: np.ndarray) -> list[Region]:
    """Detector exploratorio de contornos ovalados con tinta en su interior.

    Es heurístico, puede fallar con globos abiertos/oscuros y ocasionalmente
    proponer elementos que no son globos. Por ello requiere revisión visual.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ink = (gray < 92).astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, cw, ch = cv2.boundingRect(contour)
        if area < max(700, h * w * 0.0015) or area > h * w * 0.18:
            continue
        if x < 5 or y < 5 or x + cw > w - 5 or y + ch > h - 5:
            continue
        if cw < 38 or ch < 42 or not 0.3 < cw / ch < 3:
            continue
        box_fill = area / (cw * ch)
        if not 0.42 < box_fill < 0.89:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or area / hull_area < 0.76:
            continue
        inside = np.zeros((ch, cw), np.uint8)
        cv2.drawContours(inside, [contour - np.array([[[x, y]]])], -1, 255, cv2.FILLED)
        inner = cv2.erode(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        if cv2.countNonZero(inner) < 400:
            continue
        black_ratio = float(
            ((gray[y : y + ch, x : x + cw] < 100) & (inner > 0)).sum()
            / max(1, cv2.countNonZero(inner))
        )
        # En el interior del globo esperamos pocos trazos en relación al fondo.
        if not 0.002 < black_ratio < 0.11:
            continue
        xa = max(0, x - 3)
        ya = max(0, y - 3)
        xb = min(w, x + cw + 3)
        yb = min(h, y + ch + 3)
        full = np.zeros((yb - ya, xb - xa), np.uint8)
        full[y - ya : y - ya + ch, x - xa : x - xa + cw] = inner
        r = bounded_region(
            image, (xa, ya, xb, yb), origin='Auto (revisar)', interior=full
        )
        safe = inner_area(r) > 0
        dark_local = gray[r.y : r.y + r.h, r.x : r.x + r.w] < 100
        if safe.sum() > 0 and float((dark_local & safe).sum() / safe.sum()) > 0.11:
            continue
        candidates.append((area, r))
    candidates.sort(key=lambda it: it[0], reverse=True)
    result = []
    for _, region in candidates:
        if any(iou_rect(region, z) > 0.6 for z in result):
            continue
        result.append(region)
    result.sort(key=lambda z: (z.y, z.x))
    for i, z in enumerate(result):
        z.id = i + 1
    return result[:35]


# Parámetros de texto_en_zona. Se ajustaron con zonas simuladas alrededor de los
# textos de las cuatro obras de desarrollo de OpenMantra. Los tamaños en píxeles
# valen para la página ya preparada (ver Pagina), no para el archivo original.
LADO_TRABAJO = 830  # px: ancho de hoja con que se busca texto (el de OpenMantra)
PAPEL_BLANCO = 250  # gris al que se lleva el papel de la página
LETRA_MAXIMA = 0.07  # del lado menor de la hoja: una pieza mayor no es una letra
BLOQUE_MAXIMO = 0.15  # de la hoja: ningún texto de desarrollo pasa de 0,10
PAPEL_CLARO = 215  # gris desde el que un píxel cuenta como papel del globo
PAPEL_MINIMO = 300  # px: zonas de papel más chicas no se revisan
TINTA_VISIBLE = 160  # piezas cuyo punto más oscuro es más claro no son tinta
TINTA_NEGRA = 110  # un grupo de letras tiene algún píxel así de oscuro…
TRAZO_GRUESO = 1.5  # …en una pieza con algún punto a esta distancia (px) de su borde
UNION_LETRAS = 1.2  # piezas más cercanas que esto (en tamaños de letra) son un bloque
MARGEN_BLOQUE = 0.25  # margen alrededor de cada bloque, en tamaños de letra
ALTO_PALABRA = 9  # px: alto mínimo de una palabra cuyas letras se tocan


def _encerrado(forma: np.ndarray) -> np.ndarray:
    """Lo que la forma (máscara booleana) encierra: lo que no se alcanza desde fuera
    de su rectángulo sin cruzarla. No incluye la forma misma."""
    libre = np.pad(~forma, 1, 'constant', constant_values=True).astype(np.uint8)
    cv2.floodFill(libre, None, (0, 0), 2)
    return (libre == 1)[1:-1, 1:-1]


def _huecos_del_papel(gris: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """Tinta que el papel rodea por completo, separada por zona de papel.

    Devuelve (x, y, huecos) por cada zona de papel con algo adentro; huecos es una
    máscara del tamaño del rectángulo de esa zona, con origen en (x, y). Lo que
    toca el borde de la imagen no es hueco: una letra cortada no se toma a medias.

    Solo cuenta la tinta que toca esa zona de papel: las letras de un globo dibujado
    sobre papel blanco son huecos del papel del globo, no del papel de afuera. Y se
    descartan los marcos, como el contorno de un globo: piezas que encierran más del
    doble de su propia área con otra tinta adentro, o más de cinco veces aunque
    esté vacía (una letra como 口 encierra solo un poco de papel).
    """
    papel = gris > PAPEL_CLARO
    # Papel con conexión 4 y tinta con conexión 8: así una línea diagonal fina
    # (el borde de un globo) separa el papel de adentro del de afuera.
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(papel.astype(np.uint8), 4)
    cruz = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    salida = []
    for k in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[k]]
        if area < PAPEL_MINIMO:
            continue
        propio = etiquetas[y : y + h, x : x + w] == k
        tinta = _encerrado(propio) & ~papel[y : y + h, x : x + w]
        if not tinta.any():
            continue
        m, piezas, pstats, _ = cv2.connectedComponentsWithStats(
            tinta.astype(np.uint8), 8
        )
        vecinas = np.zeros(m, bool)
        toca = cv2.dilate(propio.astype(np.uint8), cruz) > 0
        vecinas[np.unique(piezas[toca & tinta])] = True
        vecinas[0] = False
        for p in np.flatnonzero(vecinas):
            px, py, pw, ph, parea = [int(v) for v in pstats[p]]
            pieza = piezas[py : py + ph, px : px + pw] == p
            dentro = _encerrado(pieza)
            adentro_hay_tinta = (dentro & tinta[py : py + ph, px : px + pw]).any()
            relacion = dentro.sum() / parea
            if (adentro_hay_tinta and relacion > 2) or relacion > 5:
                vecinas[p] = False  # marco: contorno de un globo o de una viñeta
        huecos = vecinas[piezas]
        if huecos.any():
            salida.append((x, y, huecos.astype(np.uint8) * 255))
    return salida


@dataclass
class Pagina:
    """Página preparada para buscar texto, con un tono de papel y un tamaño comunes.

    Los escaneos varían: papel gris o amarillento, tinta lavada, más o menos
    resolución. Sin esto, con un papel apenas gris no había ningún «papel blanco»
    y no se encontraba nada.
    gris: gris suavizado con el papel en PAPEL_BLANCO, a la escala de trabajo.
    limpio: lo mismo sin suavizar. crudo: el gris sin aclarar, a esa escala.
    escala: píxeles de trabajo por píxel del archivo.
    """

    gris: np.ndarray
    limpio: np.ndarray
    crudo: np.ndarray
    escala: float


def nivel_de_papel(gris: np.ndarray) -> int:
    """Gris del papel: el tono claro más claro que reúne al menos un 3 % de los
    píxeles (255 si no hay ninguno desde 130, por ejemplo en una hoja negra)."""
    masa = np.bincount(gris.ravel(), minlength=256) / max(1, gris.size)
    masa = np.convolve(masa, np.ones(9), 'same')  # cada tono con sus 4 vecinos
    for nivel in range(255, 129, -1):
        if masa[nivel] >= 0.03:
            while nivel > 130 and masa[nivel - 1] > masa[nivel]:
                nivel -= 1
            return nivel
    return 255


def _estirar(gris: np.ndarray, negro: float, papel: float) -> np.ndarray:
    """Lleva el tono `negro` a 0 y el tono `papel` a PAPEL_BLANCO."""
    estirado = (
        (gris.astype(np.float32) - negro) * PAPEL_BLANCO / max(1.0, papel - negro)
    )
    return np.clip(estirado, 0, 255).astype(np.uint8)


def _negro_lavado(gris: np.ndarray) -> float:
    """Tono de la tinta si está lavada (más clara que 40); si no, 0.

    Se mira el 0,05 % más oscuro: basta un texto chico con tinta negra, aunque el
    resto de la hoja sea claro, para saber que la tinta no está lavada.
    """
    negro = float(np.percentile(gris, 0.05))
    return min(negro, 90.0) if negro > 40 else 0.0


def aclarar_papel(gris: np.ndarray) -> np.ndarray:
    """Página en gris con el papel casi blanco y la tinta negra. Una página con buen
    contraste (papel desde 240 y tinta hasta 40) queda igual."""
    papel, negro = nivel_de_papel(gris), _negro_lavado(gris)
    if papel >= 240 and negro == 0:
        return gris
    return _estirar(gris, negro, papel)


def preparar_pagina(image: np.ndarray) -> Pagina:
    """Prepara la página para buscar texto (ver Pagina).

    Una hoja doble se mide por la mitad de su ancho. Las imágenes chicas (menos de
    480 px) se toman como recortes y no se agrandan.
    """
    crudo = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alto, ancho = crudo.shape
    hoja = ancho / 2 if ancho > 1.2 * alto else min(alto, ancho)
    escala = LADO_TRABAJO / hoja
    if hoja < 480 or 0.75 < escala < 1.35:
        escala = 1.0
    else:
        crudo = cv2.resize(
            crudo,
            None,
            fx=escala,
            fy=escala,
            interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC,
        )
    limpio = aclarar_papel(crudo)
    return Pagina(cv2.GaussianBlur(limpio, (3, 3), 0), limpio, crudo, escala)


def _region_de(image: np.ndarray, caja: tuple, escala: float, origin: str) -> Region:
    """Región del archivo que corresponde a una caja (x1, y1, x2, y2) de trabajo."""
    x1, y1 = int(caja[0] / escala), int(caja[1] / escala)
    x2, y2 = int(np.ceil(caja[2] / escala)), int(np.ceil(caja[3] / escala))
    return bounded_region(image, (x1, y1, x2, y2), origin=origin)


def _bloques_de_texto(gris: np.ndarray, pagina: Pagina) -> list[tuple]:
    """Bloques de texto (x1, y1, x2, y2) de un recorte de la página preparada.

    gris tiene el papel en blanco; las letras son los huecos de ese papel. Los
    huecos se agrupan por globo y por cercanía, según el tamaño de letra que se
    mide en cada globo (con un tope: si no, en una zona de papel con pocas figuras
    grandes adentro se tomaban las figuras por letras y se unía la hoja entera).
    Se descartan los grupos sin tinta negra de trazo grueso (pelos, gotas de sudor
    y líneas finas del dibujo también son huecos del papel), las piezas sueltas
    chicas o alargadas y los bloques más grandes que cualquier texto.
    """
    alto, ancho = gris.shape
    alto_p, ancho_p = pagina.gris.shape
    tope = max(40.0, LETRA_MAXIMA * min(alto_p, ancho_p))
    globos = []
    for x, y, huecos in _huecos_del_papel(gris):
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(huecos, 8)
        # Tinta más oscura y grosor máximo de cada pieza.
        sub = gris[y : y + huecos.shape[0], x : x + huecos.shape[1]]
        oscuro = np.full(n, 255.0, np.float32)
        np.minimum.at(oscuro, etiquetas.ravel(), sub.ravel().astype(np.float32))
        grosor = np.zeros(n, np.float32)
        distancia = cv2.distanceTransform(huecos, cv2.DIST_L2, 3)
        np.maximum.at(grosor, etiquetas.ravel(), distancia.ravel())
        # Como en generate_mask: muchos puntos chicos y finos son una trama, no letras.
        puntos = (stats[:, 4] < 30) & (grosor < GROSOR_LETRA)
        if int(puntos[1:].sum()) < PUNTOS_DE_TRAMA:
            puntos[:] = False
        tinta = (stats[:, 4] >= 2) & (oscuro <= TINTA_VISIBLE) & ~puntos
        tinta[0] = False
        lado = np.maximum(stats[:, 2], stats[:, 3])
        globos.append((x, y, etiquetas, stats, oscuro, grosor, tinta, lado))
    medibles = [g[7][g[6] & (g[7] >= 3)] for g in globos]
    if not medibles or not np.concatenate(medibles).size:
        return []
    letra_zona = float(np.percentile(np.concatenate(medibles), 75))
    cajas = []
    for (x, y, etiquetas, stats, oscuro, grosor, tinta, lado), propios in zip(
        globos, medibles
    ):
        # Tamaño de letra: el de este globo si hay piezas suficientes para medirlo.
        letra = float(np.percentile(propios, 75)) if len(propios) >= 4 else letra_zona
        letra = min(letra, tope)
        sirve = tinta & (lado <= 3 * letra)
        de_letra = sirve & (oscuro <= TINTA_NEGRA) & (grosor >= TRAZO_GRUESO)
        piezas = np.where(sirve[etiquetas], 255, 0).astype(np.uint8)
        if not piezas.any():
            continue
        union = max(3, int(round(UNION_LETRAS * letra)))
        bloques = cv2.connectedComponents(
            cv2.dilate(
                piezas, cv2.getStructuringElement(cv2.MORPH_RECT, (union, union))
            ),
            connectivity=8,
        )[1]
        bloques[piezas == 0] = 0
        margen = max(3, int(round(MARGEN_BLOQUE * letra)))
        for b in np.unique(bloques)[1:]:
            ys, xs = np.nonzero(bloques == b)
            ids = np.unique(etiquetas[ys, xs])
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            largo, corto = max(x2 - x1, y2 - y1), min(x2 - x1, y2 - y1)
            if largo < 0.5 * letra or not de_letra[ids].any():
                continue
            # Una pieza suelta: una mota, una línea del dibujo… o una palabra corta
            # en letra latina, cuyas letras se tocan («HEY»): alargada, pero con el
            # alto de una letra.
            if len(ids) == 1 and (
                largo < 0.8 * letra
                or (largo > 3 * corto and (corto < ALTO_PALABRA or largo > 8 * corto))
            ):
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > AREA_REVISAR and area > BLOQUE_MAXIMO * alto_p * ancho_p:
                continue  # más grande que cualquier texto: viñeta o dibujo
            cajas.append(
                (
                    max(0, x + x1 - margen),
                    max(0, y + y1 - margen),
                    min(ancho, x + x2 + margen),
                    min(alto, y + y2 + margen),
                )
            )
    return cajas


def texto_en_zona(
    image: np.ndarray,
    xyxy: tuple[int, int, int, int],
    pagina: Optional[Pagina] = None,
) -> list[Region]:
    """Bloques de texto dentro de una zona marcada a grandes rasgos.

    El usuario no necesita ajustar el rectángulo a las letras: basta con encerrar
    el globo (o varios). Las letras de un globo son los «huecos» de su papel: lo
    oscuro que el papel blanco rodea por completo. Por eso no importa lo que haya
    alrededor del globo (fondo negro, tramas, dibujo), que en cambio desvía el
    umbral de Otsu calculado sobre la zona entera y deja fuera las letras finas.

    Cada bloque se devuelve como una región ajustada, con un margen de un cuarto de
    letra. Si con el tono de papel de la página no aparece nada, se prueba con el
    de la zona (un globo gris, una sombra del escaneo). Si no hay texto en globos
    (por ejemplo, texto sobre el dibujo), la lista queda vacía.
    """
    zona = bounded_region(image, xyxy)
    pagina = preparar_pagina(image) if pagina is None else pagina
    e = pagina.escala
    x0, y0 = int(zona.x * e), int(zona.y * e)
    corte = (
        slice(y0, int(np.ceil((zona.y + zona.h) * e))),
        slice(x0, int(np.ceil((zona.x + zona.w) * e))),
    )
    cajas = _bloques_de_texto(pagina.gris[corte], pagina)
    if not cajas and pagina.gris[corte].size < pagina.gris.size:
        propio = cv2.GaussianBlur(aclarar_papel(pagina.crudo[corte]), (3, 3), 0)
        cajas = _bloques_de_texto(propio, pagina)
    regiones = []
    for x1, y1, x2, y2 in cajas:
        try:
            r = _region_de(image, (x0 + x1, y0 + y1, x0 + x2, y0 + y2), e, 'Zona')
            # Con el redondeo de la escala no debe salirse de la zona pedida.
            regiones.append(
                bounded_region(
                    image,
                    (
                        max(r.x, zona.x),
                        max(r.y, zona.y),
                        min(r.x + r.w, zona.x + zona.w),
                        min(r.y + r.h, zona.y + zona.h),
                    ),
                    origin='Zona',
                )
            )
        except ValueError:
            continue  # bloque de menos de 12 × 12 px
    regiones.sort(key=lambda r: (r.y, r.x))
    return regiones


# ---------------------------------------------------------------------------
# Detector de texto entrenado
# ---------------------------------------------------------------------------
# Los candidatos son los bloques que texto_en_zona encuentra en la página
# entera: aparece casi todo el texto, pero también trozos de dibujo encerrados
# por papel blanco. Cada candidato se describe con las medidas de RASGOS_TEXTO
# y un random forest de OpenCV (cv2.ml.RTrees) da la probabilidad de que sea
# texto. El modelo se entrena con `python -m evaluacion.entrenar_detector`
# (OpenMantra sin la obra de prueba, más las etiquetas hechas a mano con
# `evaluacion.etiquetar --texto`) y se guarda en modelo_texto.yml.

RASGOS_TEXTO = (
    'piezas',  # log(1 + piezas de tinta)
    'cv_lado',  # cuánto varía el tamaño de las piezas (las letras se parecen)
    'tinta',  # fracción del rectángulo con tinta
    'grosor_rel',  # grosor típico del trazo, relativo al tamaño de letra
    'cv_grosor',  # cuánto varía el grosor (el de las letras es parejo)
    'oscuro',  # percentil 10 del gris de la tinta (las letras son negras)
    'papel_anillo',  # fracción de papel blanco alrededor del bloque
    'bordes_anillo',  # densidad de bordes alrededor (el dibujo tiene muchos)
    'alargado',  # |log(ancho / alto)|: columnas o renglones de texto
    'area_rel',  # log(área / área de la página)
    'letra_rel',  # tamaño de letra / alto de la página
    'en_globo',  # fracción dentro del interior de un globo de detect_bubbles
    'huecos_max',  # huecos entre columnas o renglones (el mayor de los dos)
    'huecos_min',  # y el menor: medidos así, no importa la orientación del texto
    'relleno',  # cuánto llena cada pieza su rectángulo
    'borde_pagina',  # distancia al borde de la página, relativa a su lado menor
    'densidad',  # piezas por «casilla» de letra
)
MODELO_TEXTO = Path(__file__).resolve().parent / 'modelo_texto.yml'
# Cambia cuando cambian los candidatos o sus medidas: las guardadas antes ya no sirven.
VERSION_RASGOS = 2
# Gris de la tinta al medir: más estricto que PAPEL_CLARO, así las letras latinas,
# que van muy juntas, no se funden en una sola pieza por su borde suavizado.
TINTA_PIEZA = 150
# Umbrales elegidos con la validación cruzada por obra (ver evaluacion/README.md).
CONFIANZA_LIMPIAR = 0.5  # desde aquí el modo automático limpia la región
CONFIANZA_PROPONER = 0.3  # desde aquí se propone; por debajo de LIMPIAR, para revisar


def interior_de_globos(image: np.ndarray) -> np.ndarray:
    """Máscara (0 o 1) del interior de los globos que encuentra detect_bubbles."""
    interior = np.zeros(image.shape[:2], np.float32)
    for g in detect_bubbles(image):
        if g.interior is not None:
            zona = interior[g.y : g.y + g.h, g.x : g.x + g.w]
            np.maximum(zona, (g.interior > 0).astype(np.float32), out=zona)
    return interior


def rasgos_de_texto(
    gris: np.ndarray, caja: tuple[int, int, int, int], interior: np.ndarray
) -> Optional[list[float]]:
    """Medidas de RASGOS_TEXTO para el rectángulo (x1, y1, x2, y2) de la página.

    gris es la página en gris suavizada (como en texto_en_zona) e interior la de
    interior_de_globos. Devuelve None si el rectángulo no tiene tinta.
    """
    alto_p, ancho_p = gris.shape
    x1, y1, x2, y2 = caja
    w, h = x2 - x1, y2 - y1
    sub = gris[y1:y2, x1:x2]
    tinta = (sub <= TINTA_PIEZA).astype(np.uint8)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(tinta, 8)
    validas = stats[1:, 4] >= 2
    lados = np.maximum(stats[1:, 2], stats[1:, 3])[validas]
    if lados.size == 0:
        return None
    medibles = lados[lados >= 3]
    letra = float(np.percentile(medibles if medibles.size else lados, 75))
    grosor = np.zeros(n, np.float32)
    distancia = cv2.distanceTransform(tinta, cv2.DIST_L2, 3)
    np.maximum.at(grosor, etiquetas.ravel(), distancia.ravel())
    grosor = grosor[1:][validas]
    # Anillo de medio tamaño de letra alrededor del bloque.
    m = max(3, int(round(0.5 * letra)))
    xa, ya = max(0, x1 - m), max(0, y1 - m)
    xb, yb = min(ancho_p, x2 + m), min(alto_p, y2 + m)
    anillo = np.ones((yb - ya, xb - xa), bool)
    anillo[y1 - ya : y2 - ya, x1 - xa : x2 - xa] = False
    entorno = gris[ya:yb, xa:xb]
    bordes = cv2.Canny(entorno, 75, 170) > 0
    oscuros = sub[tinta > 0]
    huecos_x = float((tinta.sum(axis=0) == 0).mean())
    huecos_y = float((tinta.sum(axis=1) == 0).mean())
    cajas_piezas = np.maximum(1, stats[1:, 2] * stats[1:, 3])[validas]
    return [
        float(np.log1p(lados.size)),
        float(lados.std() / max(1e-6, lados.mean())),
        float(tinta.mean()),
        float(np.median(grosor) / max(1.0, letra)),
        float(grosor.std() / max(1e-6, grosor.mean())),
        float(np.percentile(oscuros, 10)) / 255,
        float((entorno[anillo] > PAPEL_CLARO).mean()) if anillo.any() else 0.0,
        float(bordes[anillo].mean()) if anillo.any() else 0.0,
        abs(float(np.log(w / h))),
        float(np.log(w * h / (alto_p * ancho_p))),
        letra / alto_p,
        float(interior[y1:y2, x1:x2].mean()),
        max(huecos_x, huecos_y),
        min(huecos_x, huecos_y),
        float(np.mean(stats[1:, 4][validas] / cajas_piezas)),
        min(x1, y1, ancho_p - x2, alto_p - y2) / min(alto_p, ancho_p),
        float(lados.size * letra * letra / (w * h)),
    ]


def candidatos_de_texto(
    image: np.ndarray, pagina: Optional[Pagina] = None
) -> tuple[list[Region], np.ndarray]:
    """Bloques de texto de la página entera y sus medidas (una fila cada uno).

    Las medidas se toman en la página preparada, así no dependen del tono ni de la
    resolución del archivo.
    """
    pagina = preparar_pagina(image) if pagina is None else pagina
    interior = interior_de_globos(cv2.cvtColor(pagina.limpio, cv2.COLOR_GRAY2BGR))
    pares = []
    for caja in _bloques_de_texto(pagina.gris, pagina):
        medidas = rasgos_de_texto(pagina.gris, caja, interior)
        if medidas is None:
            continue
        try:
            pares.append((_region_de(image, caja, pagina.escala, 'Detectada'), medidas))
        except ValueError:
            continue  # bloque de menos de 12 × 12 px
    pares.sort(key=lambda par: (par[0].y, par[0].x))
    medidas = np.array([m for _, m in pares], np.float32)
    return [r for r, _ in pares], medidas.reshape(-1, len(RASGOS_TEXTO))


_modelos: dict = {}


def cargar_modelo(ruta: Path = MODELO_TEXTO):
    """Random forest guardado (o None si no existe). Se lee el texto con Python y
    se pasa a OpenCV en memoria: así funciona aunque la ruta tenga tildes."""
    ruta = Path(ruta)
    if ruta not in _modelos:
        modelo = None
        if ruta.exists():
            fs = cv2.FileStorage(
                ruta.read_text(encoding='utf8'),
                cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY,
            )
            modelo = cv2.ml.RTrees_create()
            modelo.read(fs.root())
        _modelos[ruta] = modelo
    return _modelos[ruta]


def probabilidad_de_texto(modelo, medidas: np.ndarray) -> np.ndarray:
    """Fracción de árboles del bosque que votan «texto» para cada fila de medidas."""
    if len(medidas) == 0:
        return np.zeros(0)
    votos = modelo.getVotes(np.asarray(medidas, np.float32), 0)
    clases = list(votos[0])
    cuentas = votos[1:].astype(np.float64)
    if 1 not in clases:
        return np.zeros(len(medidas))
    return cuentas[:, clases.index(1)] / np.maximum(1, cuentas.sum(axis=1))


def _zonas_de_papel(gris: np.ndarray, area_maxima: float) -> np.ndarray:
    """Etiqueta las zonas de papel de hasta area_maxima px (0 = tinta o papel mayor)."""
    _, etiquetas, stats, _ = cv2.connectedComponentsWithStats(
        (gris > PAPEL_CLARO).astype(np.uint8), connectivity=4
    )
    grandes = stats[:, 4] > area_maxima
    grandes[0] = True
    etiquetas[grandes[etiquetas]] = 0
    return etiquetas


def papel_de_pagina(image: np.ndarray, pagina: Optional[Pagina] = None) -> np.ndarray:
    """Etiqueta, en píxeles del archivo, cada zona de papel del tamaño de un globo
    (0 = tinta, o papel que ocupa más de FRACCION_PAGINA_REVISAR de la página,
    como el fondo blanco alrededor de las viñetas)."""
    pagina = preparar_pagina(image) if pagina is None else pagina
    gris = pagina.gris
    etiquetas = _zonas_de_papel(gris, FRACCION_PAGINA_REVISAR * gris.size)
    if pagina.escala == 1:
        return etiquetas
    alto, ancho = image.shape[:2]
    filas = np.minimum((np.arange(alto) * pagina.escala).astype(int), gris.shape[0] - 1)
    cols = np.minimum((np.arange(ancho) * pagina.escala).astype(int), gris.shape[1] - 1)
    return etiquetas[np.ix_(filas, cols)]


def _papel_alrededor(papel: np.ndarray, caja: tuple[int, int, int, int]) -> int:
    """Zona de papel que más rodea a la caja (x1, y1, x2, y2), en una franja de 3 px
    a su alrededor, o 0."""
    alto, ancho = papel.shape
    x1, y1, x2, y2 = caja
    xa, ya = max(0, x1 - 3), max(0, y1 - 3)
    franja = papel[ya : min(alto, y2 + 3), xa : min(ancho, x2 + 3)].copy()
    franja[y1 - ya : y2 - ya, x1 - xa : x2 - xa] = 0
    cuentas = np.bincount(franja.ravel())
    cuentas[0] = 0
    return int(cuentas.argmax()) if cuentas.any() else 0


def puntuar_candidatos(
    image: np.ndarray,
    modelo=None,
    papel: Optional[np.ndarray] = None,
    pagina: Optional[Pagina] = None,
) -> list[Region]:
    """Todos los candidatos de la página, cada uno con su confianza de ser texto
    y la zona de papel (globo) que lo rodea.

    Sin modelo entrenado, todos quedan en CONFIANZA_PROPONER: se proponen pero
    ninguno se limpia solo.
    """
    pagina = preparar_pagina(image) if pagina is None else pagina
    papel = papel_de_pagina(image, pagina) if papel is None else papel
    regiones, medidas = candidatos_de_texto(image, pagina)
    modelo = modelo if modelo is not None else cargar_modelo()
    if modelo is None:
        probabilidades = np.full(len(regiones), CONFIANZA_PROPONER)
    else:
        probabilidades = probabilidad_de_texto(modelo, medidas)
    for r, p in zip(regiones, probabilidades):
        r.confianza = float(p)
        r.papel = _papel_alrededor(papel, (r.x, r.y, r.x + r.w, r.y + r.h))
    return regiones


def demasiado_grande(ancho: int, alto: int, forma_pagina: tuple[int, ...]) -> bool:
    """Indica si un rectángulo es más grande que el texto habitual de una página:
    más de AREA_REVISAR px y de FRACCION_PAGINA_REVISAR de la página."""
    area = ancho * alto
    limite = FRACCION_PAGINA_REVISAR * forma_pagina[0] * forma_pagina[1]
    return area > AREA_REVISAR and area > limite


def _distancia(region: Region, punto: tuple[int, int]) -> float:
    """Distancia en píxeles del punto al rectángulo de la región (0 si está adentro)."""
    x, y = punto
    dx = max(region.x - x, 0, x - (region.x + region.w - 1))
    dy = max(region.y - y, 0, y - (region.y + region.h - 1))
    return float(np.hypot(dx, dy))


def regiones_del_punto(
    regiones: list[Region], punto: tuple[int, int], papel: np.ndarray
) -> list[Region]:
    """Regiones que confirma un punto marcado por el usuario.

    Son las que están en el mismo globo que el punto (según papel_de_pagina) y, si
    no hay ninguna, la más cercana que lo contenga o esté a pocos píxeles. Nunca
    una región demasiado grande para ser texto: una que cubre media hoja contiene
    casi cualquier punto, y marcar uno no significa que todo eso sea texto.
    """
    x, y = punto
    posibles = [r for r in regiones if not demasiado_grande(r.w, r.h, papel.shape)]
    dentro = 0 <= y < papel.shape[0] and 0 <= x < papel.shape[1]
    globo = int(papel[y, x]) if dentro else 0
    if globo:
        del_globo = [r for r in posibles if r.papel == globo]
        if del_globo:
            return del_globo
    cerca = [
        (_distancia(r, punto), r.w * r.h, k)
        for k, r in enumerate(posibles)
        if _distancia(r, punto) <= max(8, 0.25 * min(r.w, r.h))
    ]
    if not cerca:
        return []
    elegida = posibles[min(cerca)[2]]
    if elegida.papel:  # las demás del mismo globo también son texto
        return [r for r in posibles if r.papel == elegida.papel]
    return [elegida]


def elegir_texto(
    candidatos: list[Region],
    puntos: tuple[tuple[int, int], ...] = (),
    papel: Optional[np.ndarray] = None,
) -> list[Region]:
    """Candidatos que se proponen como regiones, como copias nuevas.

    Un punto marcado por el usuario confirma (confianza 1) el texto de su globo
    (ver regiones_del_punto; hace falta papel). Los candidatos sin un punto no
    pierden nada: se proponen si su confianza llega a CONFIANZA_PROPONER.
    """
    confirmados = {
        id(r) for p in puntos for r in regiones_del_punto(candidatos, p, papel)
    }
    elegidos = []
    for c in candidatos:
        confianza = 1.0 if id(c) in confirmados else c.confianza
        if confianza >= CONFIANZA_PROPONER:
            elegidos.append(replace(c, confianza=confianza))
    return elegidos


def buscar_en_punto(
    image: np.ndarray, punto: tuple[int, int], pagina: Optional[Pagina] = None
) -> list[Region]:
    """Texto alrededor de un punto que marcó el usuario y que el detector no propuso.

    Busca en una ventana alrededor del punto, tomando como papel el tono que hay
    justo ahí: así sirve en un globo gris o en una parte sombreada del escaneo.
    Devuelve los bloques del mismo globo que el punto o, si el papel del punto es
    muy grande (no es un globo), el bloque que esté a menos de 40 px de trabajo.
    """
    pagina = preparar_pagina(image) if pagina is None else pagina
    crudo, e = pagina.crudo, pagina.escala
    alto, ancho = crudo.shape
    px = min(ancho - 1, max(0, int(punto[0] * e)))
    py = min(alto - 1, max(0, int(punto[1] * e)))
    parche = crudo[max(0, py - 12) : py + 13, max(0, px - 12) : px + 13]
    tono = float(np.percentile(parche, 90))
    if tono < 90:
        return []  # fondo negro: el texto blanco sobre negro no se detecta
    lado = int(0.3 * min(alto, ancho))
    x0, y0 = max(0, px - lado), max(0, py - lado)
    ventana = crudo[y0 : py + lado, x0 : px + lado]
    gris = cv2.GaussianBlur(_estirar(ventana, _negro_lavado(ventana), tono), (3, 3), 0)
    zonas = _zonas_de_papel(gris, FRACCION_PAGINA_REVISAR * crudo.size)
    # Papel del punto: el más repetido en el parche (el punto puede caer en una letra).
    vx, vy = px - x0, py - y0
    cuentas = np.bincount(
        zonas[max(0, vy - 12) : vy + 13, max(0, vx - 12) : vx + 13].ravel()
    )
    cuentas[0] = 0
    globo = int(cuentas.argmax()) if cuentas.any() else 0
    regiones, sueltas = [], []
    for caja in _bloques_de_texto(gris, pagina):
        try:
            r = _region_de(
                image,
                (x0 + caja[0], y0 + caja[1], x0 + caja[2], y0 + caja[3]),
                e,
                'Punto',
            )
        except ValueError:
            continue
        if demasiado_grande(r.w, r.h, image.shape):
            continue
        if globo and _papel_alrededor(zonas, caja) == globo:
            regiones.append(r)
        elif _distancia(r, punto) * e <= 40:
            sueltas.append(r)
    if regiones or not sueltas:
        return regiones
    return [min(sueltas, key=lambda r: _distancia(r, punto))]


def detectar_texto(
    image: np.ndarray, puntos: tuple[tuple[int, int], ...] = (), modelo=None
) -> list[Region]:
    """Detector de texto de la página: candidatos puntuados y, si los hay, los
    puntos del usuario. Las regiones con confianza menor que CONFIANZA_LIMPIAR
    quedan para revisión (ver describe_background)."""
    pagina = preparar_pagina(image)
    papel = papel_de_pagina(image, pagina)
    candidatos = puntuar_candidatos(image, modelo, papel, pagina)
    return elegir_texto(candidatos, tuple(puntos), papel)


def iou_rect(a: Region, b: Region) -> float:
    """Intersección sobre unión (IoU) de los rectángulos de dos regiones, de 0 a 1."""
    xa = max(a.x, b.x)
    ya = max(a.y, b.y)
    xb = min(a.x + a.w, b.x + b.w)
    yb = min(a.y + a.h, b.y + b.h)
    intersect = max(0, xb - xa) * max(0, yb - ya)
    union = a.w * a.h + b.w * b.h - intersect
    return float(intersect / union) if union else 0.0


def inner_area(region: Region) -> np.ndarray:
    """Zona de la región que se puede modificar: el interior del globo (o todo el
    rectángulo) erosionado unos píxeles para no tocar el contorno.

    La erosión solo achica el interior de un globo detectado: en una selección
    manual el rectángulo entero queda permitido (OpenCV no erosiona desde el
    borde de la imagen), así una selección ajustada no deja sin borrar el borde
    de las letras. Lo que entra desde fuera lo descarta _tinta_que_sigue_afuera.
    """
    if region.interior is not None:
        mask = region.interior.copy()
    else:
        mask = np.full((region.h, region.w), 255, np.uint8)
    margin = max(2, min(5, int(min(region.w, region.h) * 0.04)))
    mask = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)),
    )
    return mask


def _line_protection(binary: np.ndarray) -> np.ndarray:
    """Identifica trazos largos horizontales/verticales (marcos y contornos).

    Es deliberadamente conservador: una linea larga puede ser arte o el borde
    del globo. En la beta preferimos que el usuario la marque si es texto.
    Fuera de la región se cuenta papel: si no, OpenCV toma el exterior como
    tinta y cualquier letra que toque el borde de la selección parecería una
    línea larga. Las líneas que siguen fuera de la selección las descarta
    _tinta_que_sigue_afuera.
    """
    h, w = binary.shape

    def abrir(ancho, alto):
        return cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (ancho, alto)),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    horizontal = abrir(max(28, int(w * 0.67)), 1)
    vertical = abrir(1, max(34, int(h * 0.70)))
    protected = cv2.bitwise_or(horizontal, vertical)
    return cv2.dilate(protected, np.ones((5, 5), np.uint8))


# Parámetros de la máscara. Se ajustaron con pares sintéticos de desarrollo
# (semilla 7), nunca con los de la evaluación (semilla 2026).
PUNTOS_DE_TRAMA = 20  # cuántos puntos pequeños y finos indican una trama de fondo
GROSOR_LETRA = (
    2.0  # distancia al borde (px) que alcanzan los trazos de letra, no la trama
)
CONTRASTE_TEXTO = (
    12  # cuánto más oscuro que su entorno debe ser un píxel para agrandar la máscara
)


def _disco(diametro: int) -> np.ndarray:
    """Elemento estructurante circular del diámetro dado (impar)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diametro, diametro))


def _forma_de_letra(
    ancho: int, alto: int, area: int, region: Region, min_area: int, max_area: float
) -> bool:
    """Indica si una componente tiene tamaño y forma de letra (no de línea ni de borde)."""
    if not min_area <= area <= max_area:
        return False
    if ancho > region.w * 0.68 or alto > region.h * 0.81:
        return False
    if ancho > region.w * 0.45 and alto <= 3:
        return False
    if alto > region.h * 0.52 and ancho <= 2:
        return False
    return True


def _rescatar_letras(
    componente: np.ndarray,
    grosor: np.ndarray,
    region: Region,
    min_area: int,
    max_area: float,
) -> np.ndarray:
    """Separa letras que quedaron unidas a rayas o tramas del fondo.

    Sobre un fondo rayado, las letras que tocan una raya forman con ella una sola
    componente enorme, que el filtro de tamaño descartaría entera. Las rayas y
    los puntos de trama son finos; los trazos de letra, más gruesos. Se conservan
    los «núcleos» (píxeles a GROSOR_LETRA o más del borde), se les devuelve su
    ancho con una dilatación limitada a la componente y las piezas resultantes
    pasan otra vez el filtro de forma.
    """
    nucleos = (componente & (grosor >= GROSOR_LETRA)).astype(np.uint8) * 255
    salida = np.zeros(componente.shape, np.uint8)
    if not nucleos.any():
        return salida
    piezas = cv2.bitwise_and(
        cv2.dilate(nucleos, _disco(5)), componente.astype(np.uint8) * 255
    )
    num, etiquetas, stats, _ = cv2.connectedComponentsWithStats(piezas, 8)
    for k in range(1, num):
        _, _, ancho, alto, area = [int(v) for v in stats[k]]
        if _forma_de_letra(ancho, alto, area, region, min_area, max_area):
            salida[etiquetas == k] = 255
    return salida


def _expandir(gris: np.ndarray, mascara: np.ndarray, pixeles: int) -> np.ndarray:
    """Agranda la máscara para cubrir el borde suavizado de las letras.

    Solo se agregan píxeles más oscuros que su entorno, medido con un black-hat
    (cierre morfológico − imagen): así se cubre el antialias de las letras sin
    repintar papel ni fondo sano alrededor.
    """
    if pixeles <= 0 or not mascara.any():
        return mascara
    agrandada = cv2.dilate(mascara, _disco(2 * pixeles + 1))
    # El elemento del black-hat debe ser más ancho que los trazos de letra.
    medio_trazo = float(
        np.percentile(cv2.distanceTransform(mascara, cv2.DIST_L2, 3)[mascara > 0], 90)
    )
    lado = max(9, int(4 * medio_trazo) | 1)
    black_hat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, _disco(lado))
    return np.where((black_hat > CONTRASTE_TEXTO) | (mascara > 0), agrandada, 0).astype(
        np.uint8
    )


def _contorno_blanco(
    gris: np.ndarray, mascara: np.ndarray, permitido: np.ndarray
) -> np.ndarray:
    """Contorno blanco de las letras escritas sobre dibujo, si lo hay.

    El texto sobre dibujo suele llevar un borde blanco que también hay que borrar.
    Se busca solo cuando el entorno de las letras es oscuro (si no, el «borde»
    sería simplemente el papel del globo): píxeles muy claros a 4 px o menos de
    las letras.
    """
    if not mascara.any():
        return np.zeros(mascara.shape, bool)
    cerca = (cv2.dilate(mascara, _disco(9)) > 0) & (mascara == 0)
    lejos = (
        (cv2.dilate(mascara, _disco(21)) > 0)
        & (cv2.dilate(mascara, _disco(13)) == 0)
        & (permitido > 0)
    )
    if not lejos.any() or np.median(gris[lejos]) >= 190:
        return np.zeros(mascara.shape, bool)
    return cerca & (gris > 215)


def _tinta_que_sigue_afuera(
    image: np.ndarray, region: Region, umbral: float, margen: int = 12
) -> np.ndarray:
    """Tinta de la región que pertenece a trazos que están sobre todo FUERA de ella.

    Cuando el rectángulo corta una sombra, el borde del globo o un trozo de
    dibujo, esa pieza entra por el borde de la selección pero sigue afuera; si se
    rellenara, quedaría un parche con el borde recto del rectángulo. Una letra
    cortada por una selección ajustada, en cambio, queda casi toda adentro.
    Se mira la tinta (gris ≤ umbral) en una ventana `margen` px más grande que la
    región y se marcan las piezas con más píxeles afuera que adentro.
    """
    alto_img, ancho_img = image.shape[:2]
    x0, y0 = max(0, region.x - margen), max(0, region.y - margen)
    x1 = min(ancho_img, region.x + region.w + margen)
    y1 = min(alto_img, region.y + region.h + margen)
    gris = cv2.GaussianBlur(
        cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (3, 3), 0
    )
    n, etiquetas = cv2.connectedComponents((gris <= umbral).astype(np.uint8), 8)
    dentro = np.zeros(etiquetas.shape, bool)
    ys = slice(region.y - y0, region.y - y0 + region.h)
    xs = slice(region.x - x0, region.x - x0 + region.w)
    dentro[ys, xs] = True
    area_dentro = np.bincount(etiquetas[dentro], minlength=n)
    area_fuera = np.bincount(etiquetas[~dentro], minlength=n)
    afuera = area_fuera > area_dentro
    # También lo que llega al borde de la ventana, es decir, que sigue más de
    # `margen` px fuera de la selección: un trazo de dibujo que la cruza. Una
    # letra cortada por una selección ajustada sobresale mucho menos. No cuentan
    # los lados donde la ventana topa con el borde de la página.
    exterior = np.zeros(etiquetas.shape, bool)
    exterior[0, :] |= y0 == region.y - margen
    exterior[-1, :] |= y1 == region.y + region.h + margen
    exterior[:, 0] |= x0 == region.x - margen
    exterior[:, -1] |= x1 == region.x + region.w + margen
    afuera |= np.bincount(etiquetas[exterior], minlength=n) > 0
    afuera[0] = False  # etiqueta 0 = papel
    return afuera[etiquetas[ys, xs]]


def _sin_piezas_del_borde(mascara: np.ndarray, afuera: np.ndarray) -> np.ndarray:
    """Quita de la máscara las piezas que tocan el borde de la selección y son
    parte de un trazo que sigue fuera de ella (ver _tinta_que_sigue_afuera).

    Solo las que tocan el borde: una letra en medio de la selección se conserva
    aunque esté pegada a una raya del fondo que sí sale de ella.
    """
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, 8)
    alto, ancho = mascara.shape
    x, y, w, h = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    toca_borde = (x == 0) | (y == 0) | (x + w >= ancho) | (y + h >= alto)
    de_afuera = np.bincount(etiquetas[afuera], minlength=n) > 0
    quitar = toca_borde & de_afuera
    quitar[0] = False
    return np.where(quitar[etiquetas], 0, mascara).astype(np.uint8)


def generate_mask(image: np.ndarray, region: Region, opts: Options) -> np.ndarray:
    """Máscara de las letras dentro del rectángulo elegido por el usuario.

    Pasos: umbral (Otsu o adaptativo), componentes conectadas con filtro de
    tamaño y forma, rescate de letras unidas a rayas del fondo, descarte de
    puntos de trama, expansión hacia el antialias de las letras y, sobre dibujo,
    el contorno blanco del texto. Se excluyen los trazos largos (bordes de viñeta
    o de globo) y los que entran por el borde de la selección pero siguen sobre
    todo fuera de ella (sombras o dibujo que el rectángulo corta). No identifica
    letras por su significado: la máscara se debe revisar.
    """
    roi = image[region.y : region.y + region.h, region.x : region.x + region.w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    umbral, binary = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    if opts.threshold == 'Adaptativo':
        binary = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 11
        )
    allowed = inner_area(region)
    binary = cv2.bitwise_and(binary, allowed)

    protected = _line_protection(binary)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    protegidas = set(np.unique(labels[protected > 0]).tolist())
    # Grosor de cada componente: la mayor distancia de uno de sus píxeles al fondo.
    grosor = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    grosor_max = np.zeros(num)
    np.maximum.at(grosor_max, labels.ravel(), grosor.ravel())
    area_roi = region.w * region.h
    min_area = max(2, int(3 + (8 - opts.sensitivity) * 0.7))
    max_area = min(area_roi * 0.22, max(260, area_roi * 0.11))
    # Muchos puntos pequeños y finos indican una trama de fondo, no letras.
    puntos = (stats[1:, 4] < 30) & (grosor_max[1:] < GROSOR_LETRA)
    hay_trama = int(puntos.sum()) >= PUNTOS_DE_TRAMA

    chosen = np.zeros(binary.shape, np.uint8)
    for k in range(1, num):
        if k in protegidas:
            continue
        _, _, ancho, alto, area = [int(v) for v in stats[k]]
        if _forma_de_letra(ancho, alto, area, region, min_area, max_area):
            if hay_trama and area < 30 and grosor_max[k] < GROSOR_LETRA:
                continue  # punto de la trama
            # Tocar el borde de la región no basta para descartar una pieza:
            # con una selección ajustada, las propias letras lo tocan. Lo que se
            # descarta más abajo es lo que sigue sobre todo fuera de la región.
            chosen[labels == k] = 255
        elif area > max_area * 0.5:
            # Demasiado grande: puede ser texto unido a rayas del fondo.
            chosen |= _rescatar_letras(labels == k, grosor, region, min_area, max_area)

    afuera = _tinta_que_sigue_afuera(image, region, umbral)
    chosen = _sin_piezas_del_borde(chosen, afuera)
    letras = chosen > 0
    blanco = _contorno_blanco(gray, chosen, allowed)
    chosen = _expandir(gray, chosen, opts.expansion)
    chosen[blanco] = 255
    chosen = cv2.bitwise_and(chosen, allowed)
    # La expansión tampoco puede crecer sobre lo que sigue afuera.
    chosen[(protected > 0) | (afuera & ~letras)] = 0
    region.mask = chosen
    region.mask_edited = False
    describe_background(image, region)
    return chosen


def _zona_de_fondo(region: Region, mask: np.ndarray) -> np.ndarray:
    """Píxeles de fondo que rodean a las letras: hasta 14 px de la máscara,
    dejando 3 px de separación para no medir el borde de los trazos.
    """
    allowed = inner_area(region) > 0
    exclusion = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
    if cv2.countNonZero(mask):
        nearest = cv2.dilate(mask, _disco(29)) > 0
        valid = allowed & nearest & ~exclusion
    else:
        valid = allowed.copy()
    if int(valid.sum()) < 40:
        valid = allowed & ~exclusion
    if int(valid.sum()) < 40:
        valid = allowed
    return valid


def _entropia(valores: np.ndarray) -> float:
    """Entropía (bits) del histograma de grises en 32 niveles de 8 tonos cada uno.

    Papel liso: casi todos los píxeles caen en uno o dos niveles (≈ 0-1 bit).
    Trama o dibujo: los tonos se reparten en muchos niveles (≈ 2-5 bits).
    """
    if not valores.size:
        return 0.0
    p = np.bincount(valores.astype(np.int64) // 8, minlength=32) / valores.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _textura_repartida(gray: np.ndarray, zona: np.ndarray) -> float:
    """Fracción de la zona donde el gris varía en una ventana de 7 × 7 px
    (desviación estándar local mayor que 10).

    Distingue una trama rala pero repartida, que varía en casi toda la zona,
    de un papel liso cruzado por una sola línea, que solo varía cerca de ella.
    El histograma (y por tanto la entropía) de ambos casos puede ser igual.
    """
    if not zona.any():
        return 0.0
    g = gray.astype(np.float32)
    media = cv2.blur(g, (7, 7))
    varianza = cv2.blur(g * g, (7, 7)) - media * media
    return float((np.sqrt(np.maximum(varianza, 0))[zona] > 10).mean())


def describe_background(image: np.ndarray, region: Region) -> tuple[str, float, float]:
    """Mide el fondo próximo a las letras (no toda la viñeta) y lo clasifica.

    Calcula, sin contar los trazos marcados ni una pequeña separación a su
    alrededor, la entropía, el tono oscuro (percentil 10) y la textura repartida
    que usa el selector, además de la varianza y la densidad de bordes que se
    informan en el reporte. Devuelve (clase, entropía, tono oscuro).
    """
    roi = image[region.y : region.y + region.h, region.x : region.x + region.w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = region.mask if region.mask is not None else np.zeros(gray.shape, np.uint8)
    valid = _zona_de_fondo(region, mask)
    vals = gray[valid]
    entropy = _entropia(vals)
    dark_level = float(np.percentile(vals, 10)) if vals.size else 255.0
    texture = _textura_repartida(gray, valid)
    # Varianza sin el 8 % más claro ni el 8 % más oscuro (menos sensible a manchas).
    if vals.size > 50:
        lo, hi = np.percentile(vals, [8, 92])
        core = vals[(vals >= lo) & (vals <= hi)]
        if core.size > 20:
            vals = core
    var = float(np.var(vals)) if vals.size else 0.0
    edge = cv2.Canny(gray, 75, 170) > 0
    edge_density = float(edge[valid].mean()) if valid.any() else 0.0
    allowed = inner_area(region)
    coverage = float(cv2.countNonZero(mask) / max(1, cv2.countNonZero(allowed)))
    area = region.w * region.h
    name, method, reason = classify_background(
        entropy,
        dark_level,
        coverage,
        texture,
        area=area,
        page_fraction=area / (image.shape[0] * image.shape[1]),
    )
    if method != 'Revisión manual' and region.confianza < CONFIANZA_LIMPIAR:
        # El detector no está seguro de que sea texto: no se limpia sola.
        method = 'Revisión manual'
        reason = (
            f'Puede no ser texto (confianza {region.confianza:.0%}). Si lo es, '
            'marca un punto sobre él o elige un método a mano.'
        )
    region.review_reason = reason
    region.variance = var
    region.edge_density = edge_density
    region.entropy = entropy
    region.dark_level = dark_level
    region.texture = texture
    region.classification = name
    region.suggested_method = method
    return name, entropy, dark_level


# Umbrales del selector. ENTROPIA_LISO y TONO_TINTA se calculan con
# `python -m evaluacion.calibrar_selector` a partir de pares sintéticos de
# desarrollo (semilla 7) y de las regiones reales etiquetadas; los pares de la
# evaluación (semilla 2026) no se usan. TEXTURA_REPARTIDA no cambia ninguna
# región de desarrollo (da lo mismo entre 0,6 y 0,95), así que se fija en el
# medio: solo aparta del «Uniforme» las tramas ralas que cubren toda la zona,
# como la del globo «¿QUÉ PASÓ?» de la página de ejemplo.
ENTROPIA_LISO = 2.0  # bits: por debajo, el fondo es papel liso
TONO_TINTA = (
    40  # gris: si el 10 % más oscuro del fondo baja de aquí, hay tinta de dibujo
)
TEXTURA_REPARTIDA = 0.8  # fracción de la zona con variación local: más, es una trama

# Una región muy grande casi nunca es solo texto: suele ser una viñeta entera
# (por ejemplo, un globo grande de detect_bubbles) y su máscara incluye dibujo.
# Medido en OpenMantra: estos dos límites dejan para revisión el 84 % de las
# regiones detectadas cuya máscara es sobre todo dibujo, y solo el 0,8 % de las
# cajas de texto reales. Se piden ambos para no bloquear todo en una imagen
# recortada, donde cualquier selección es una parte grande de la imagen.
FRACCION_PAGINA_REVISAR = 0.05  # de la página
AREA_REVISAR = 40_000  # píxeles


def classify_background(
    entropy: float,
    dark_level: float,
    coverage: float = 0.0,
    texture: float = 0.0,
    umbrales: Optional[tuple[float, float]] = None,
    *,
    area: int = 0,
    page_fraction: float = 0.0,
) -> tuple[str, str, str]:
    """Regla del selector: devuelve (clase, método sugerido, motivo de revisión).

    entropy (bits), dark_level (percentil 10 de gris) y texture (fracción con
    variación local) describen el fondo cercano al texto. coverage es la
    fracción del área útil que cubre la máscara, y area y page_fraction el
    tamaño de la región en píxeles y como fracción de la página; con sus valores
    por defecto se aplica solo la regla de fondo (así la usa la evaluación).
    umbrales = (entropía, tono) reemplaza a ENTROPIA_LISO y TONO_TINTA; solo lo
    usa la calibración para probar otros valores.

    - Uniforme: pocos tonos (entropía baja) y sin una trama que cubra toda la
      zona. Basta rellenar con la mediana.
    - Complejo: hay tinta casi negra cerca de las letras, como los trazos de un
      dibujo. Se deja para revisión manual.
    - Estructurado: muchos tonos pero sin negro, como una trama, un rayado fino
      o un degradado. Se reconstruye con inpainting.
    """
    # Guardarraíles: una región demasiado grande, o una máscara que cubre gran
    # parte de su zona, probablemente incluyen dibujo o bordes, no solo letras.
    if area > AREA_REVISAR and page_fraction > FRACCION_PAGINA_REVISAR:
        return (
            'Revisar máscara',
            'Revisión manual',
            f'La región ocupa el {page_fraction:.0%} de la página: es grande para '
            'ser solo texto y la máscara puede incluir dibujo. Revísala o marca '
            'solo las letras.',
        )
    if coverage > 0.36:
        return (
            'Revisar máscara',
            'Revisión manual',
            f'La máscara cubre {coverage:.0%} del área útil. Revísala antes de limpiar.',
        )
    entropia_liso, tono_tinta = umbrales or (ENTROPIA_LISO, TONO_TINTA)
    if entropy < entropia_liso and texture < TEXTURA_REPARTIDA:
        return 'Uniforme', 'Relleno local', ''
    if dark_level >= tono_tinta:
        return 'Estructurado', 'Inpainting Telea', ''
    return (
        'Complejo / revisar',
        'Revisión manual',
        'Fondo detallado: revisa la máscara o elige el método manualmente.',
    )


def clean_region(
    image: np.ndarray, region: Region, opts: Options
) -> tuple[np.ndarray, str, int]:
    """Limpia los píxeles de la máscara y devuelve (imagen, método usado, píxeles limpiados).

    Con opts.method == 'Automático' usa el método sugerido por describe_background.
    Nunca modifica píxeles fuera de la máscara. Si no hay nada que limpiar o el
    método es «Revisión manual», devuelve una copia sin cambios y 0 píxeles.
    """
    if region.mask is None:
        generate_mask(image, region, opts)
    mask = region.mask
    if mask is None or cv2.countNonZero(mask) == 0:
        return image.copy(), 'Sin trazos detectados', 0
    method = region.suggested_method if opts.method == 'Automático' else opts.method
    ys, xs = slice(region.y, region.y + region.h), slice(region.x, region.x + region.w)
    roi = image[ys, xs]
    if method == 'Revisión manual':
        return image.copy(), 'Revisión manual (sin aplicar)', 0
    if method == 'Relleno local':
        dilated = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        )
        bg = (dilated > 0) & (mask == 0) & (inner_area(region) > 0)
        if int(bg.sum()) < 12:
            bg = (mask == 0) & (inner_area(region) > 0)
        if int(bg.sum()) < 12:
            return image.copy(), 'Revisión manual (sin fondo de referencia)', 0
        color = np.median(roi[bg], axis=0).astype(np.uint8)
        filled = np.broadcast_to(color, roi.shape).copy()
    elif method in ('Inpainting NS', 'Inpainting Telea'):
        algo = cv2.INPAINT_NS if method == 'Inpainting NS' else cv2.INPAINT_TELEA
        filled = cv2.inpaint(roi, mask, float(opts.radius), algo)
    else:
        raise ValueError(f'Método desconocido: {method}')
    output = image.copy()
    output[ys, xs][mask > 0] = filled[mask > 0]
    return output, method, int(cv2.countNonZero(mask))


def mask_whole(image: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Junta las máscaras de las regiones en una sola máscara del tamaño de la página."""
    mask = np.zeros(image.shape[:2], np.uint8)
    for region in regions:
        if region.mask is not None:
            ys = slice(region.y, region.y + region.h)
            xs = slice(region.x, region.x + region.w)
            mask[ys, xs] = cv2.bitwise_or(mask[ys, xs], region.mask)
    return mask
