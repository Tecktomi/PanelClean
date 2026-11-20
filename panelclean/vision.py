"""Pipeline PDI: contornos, segmentación, componentes, morfología y reconstrucción."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np


@dataclass
class Options:
    """Ajustes de la máscara y de la limpieza elegidos en la interfaz."""

    threshold: str = "Otsu"
    sensitivity: int = 4
    expansion: int = 1
    radius: int = 3
    method: str = "Automático"


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
    origin: str = "Manual"
    interior: Optional[np.ndarray] = field(default=None, repr=False)
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    classification: str = "Sin analizar"
    variance: float = 0.0
    edge_density: float = 0.0
    suggested_method: str = ""
    method_override: str = "Automático"
    mask_edited: bool = False
    review_reason: str = ""
    last_method: str = ""
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
        rgb = np.asarray(ImageOps.exif_transpose(obj).convert("RGB"))
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
    origin: str = "Manual",
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
    region = Region(x1, y1, x2 - x1, y2 - y1, id, origin)
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
    """
    if region.interior is not None:
        mask = region.interior.copy()
    else:
        mask = np.full((region.h, region.w), 255, np.uint8)
    # Protección del contorno externo incluso en selección manual.
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
    """
    h, w = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(28, int(w * 0.67)), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(34, int(h * 0.70)))),
    )
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


def generate_mask(image: np.ndarray, region: Region, opts: Options) -> np.ndarray:
    """Máscara de las letras dentro del rectángulo elegido por el usuario.

    Pasos: umbral (Otsu o adaptativo), componentes conectadas con filtro de
    tamaño y forma, rescate de letras unidas a rayas del fondo, descarte de
    puntos de trama, expansión hacia el antialias de las letras y, sobre dibujo,
    el contorno blanco del texto. Los trazos largos (bordes de viñeta o de
    globo) se excluyen. No identifica letras por su significado: la máscara se
    debe revisar.
    """
    roi = image[region.y : region.y + region.h, region.x : region.x + region.w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    if opts.threshold == 'Adaptativo':
        binary = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 11
        )
    else:
        _, binary = cv2.threshold(
            smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
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
            # Las componentes que tocan el borde de la región NO se descartan,
            # porque con una selección ajustada las propias letras tocan ese borde.
            # Proteger el contorno del globo queda pendiente (tarea T1.3.4).
            chosen[labels == k] = 255
        elif area > max_area * 0.5:
            # Demasiado grande: puede ser texto unido a rayas del fondo.
            chosen |= _rescatar_letras(labels == k, grosor, region, min_area, max_area)

    blanco = _contorno_blanco(gray, chosen, allowed)
    chosen = _expandir(gray, chosen, opts.expansion)
    chosen[blanco] = 255
    chosen = cv2.bitwise_and(chosen, allowed)
    chosen[protected > 0] = 0
    region.mask = chosen
    region.mask_edited = False
    describe_background(image, region)
    return chosen


def describe_background(image: np.ndarray, region: Region) -> tuple[str, float, float]:
    """Estima el fondo próximo a letras, no toda la viñeta.

    Varianza y densidad de bordes se calculan excluyendo trazos marcados y
    una pequeña separación alrededor de la máscara.
    """
    roi = image[region.y : region.y + region.h, region.x : region.x + region.w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    allowed = inner_area(region) > 0
    mask = region.mask if region.mask is not None else np.zeros(gray.shape, np.uint8)
    exclusion = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
    if cv2.countNonZero(mask):
        nearest = (
            cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))) > 0
        )
        valid = allowed & nearest & ~exclusion
    else:
        valid = allowed.copy()
    if int(valid.sum()) < 40:
        valid = allowed & ~exclusion
    if int(valid.sum()) < 40:
        valid = allowed
    vals = gray[valid]
    if vals.size > 50:
        lo, hi = np.percentile(vals, [8, 92])
        core = vals[(vals >= lo) & (vals <= hi)]
        if core.size > 20:
            vals = core
    var = float(np.var(vals)) if vals.size else 0.0
    edge = cv2.Canny(gray, 75, 170) > 0
    edge_density = float(edge[valid].mean()) if valid.any() else 0.0
    coverage = float(cv2.countNonZero(mask) / max(1, np.count_nonzero(allowed)))
    name, method, reason = classify_background(var, edge_density, coverage)
    region.review_reason = reason
    region.variance = var
    region.edge_density = edge_density
    region.classification = name
    region.suggested_method = method
    return name, var, edge_density


def classify_background(
    variance: float, edge_density: float, coverage: float = 0.0
) -> tuple[str, str, str]:
    """Regla del selector: devuelve (clase, método sugerido, motivo de revisión).

    variance y edge_density describen el fondo cercano al texto. coverage es la
    fracción del área útil que cubre la máscara; con coverage = 0 se aplica solo
    la regla de fondo (así la usa la evaluación del selector).
    """
    # Guardarraíl: una máscara que cubre gran parte de su zona puede estar
    # incluyendo dibujo o borde, no solo letras.
    if coverage > 0.36:
        return (
            'Revisar máscara',
            'Revisión manual',
            f'La máscara cubre {coverage:.0%} del área útil. Revísala antes de limpiar.',
        )
    if variance < 120 and edge_density < 0.055:
        return 'Uniforme', 'Relleno local', ''
    if variance < 2600 and edge_density < 0.21:
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
