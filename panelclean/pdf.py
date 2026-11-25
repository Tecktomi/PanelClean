"""Páginas de un PDF, tratadas como si cada una fuera una imagen suelta.

Una página se nombra con su archivo, «#» y su número, contado desde 1: la ruta
«C:\\mangas\\tomo_01.pdf#003» es la tercera página de tomo_01.pdf. Así la usan la cola
de hojas de la aplicación y las cajas marcadas a mano (evaluacion/cajas.py).

En un manga escaneado, cada página es una sola imagen: se lee tal como está guardada,
en su tamaño y sin los márgenes que le agrega el PDF. Si la página tiene otra cosa
(texto, dibujos, varias imágenes), se dibuja completa con la resolución de su imagen
principal.

Se lee con pypdfium2 (PDFium), que se importa recién al abrir un PDF: sin esa
biblioteca, todo lo demás sigue funcionando.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

_PAGINA = re.compile(r'(.+\.pdf)#(\d+)', re.IGNORECASE | re.DOTALL)
PUNTOS_POR_PULGADA = 72  # unidad de medida de un PDF
PPP = 200  # resolución de una página sin imágenes
LADO_MAXIMO = 5000  # px: lado mayor de una página dibujada


def es_pdf(ruta) -> bool:
    """Indica si la ruta es la de un PDF completo (no la de una de sus páginas)."""
    return Path(ruta).suffix.lower() == '.pdf'


def separar(ruta) -> tuple[Path, int] | None:
    """(PDF, número de página) si la ruta es una página de un PDF; si no, None."""
    partes = _PAGINA.fullmatch(str(ruta))
    return (Path(partes[1]), int(partes[2])) if partes else None


def nombre_base(ruta) -> str:
    """Nombre sin extensión, para ponerle a un archivo: «tomo_01_p003» si es de un PDF."""
    if separar(ruta) is None:
        return Path(ruta).stem
    archivo, numero = str(ruta).rsplit('#', 1)
    return f'{Path(archivo).stem}_p{numero}'


def paginas(ruta) -> list[Path]:
    """Rutas de todas las páginas de un PDF, en orden."""
    documento = _abrir(ruta)
    try:
        total = len(documento)
    finally:
        documento.close()
    cifras = max(3, len(str(total)))
    return [Path(f'{ruta}#{n:0{cifras}}') for n in range(1, total + 1)]


def elegir(texto: str, total: int) -> list[int]:
    """Números de página de un texto como «1-20» o «3, 8, 12-15», sin repetir.

    Lanza ValueError si no se entiende o si pide páginas que el PDF no tiene.
    """
    numeros = []
    for parte in texto.replace(';', ',').split(','):
        rango = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*)?', parte)
        desde, hasta = (int(rango[1]), int(rango[2] or rango[1])) if rango else (0, 0)
        if not 1 <= desde <= hasta <= total:
            raise ValueError(
                f'«{parte.strip()}» no sirve: escribe páginas entre 1 y {total}, '
                'por ejemplo 1-20 o 3, 8, 12-15.'
            )
        numeros += [n for n in range(desde, hasta + 1) if n not in numeros]
    return numeros


def leer(ruta, numero: int) -> np.ndarray:
    """Página de un PDF como imagen BGR."""
    documento = _abrir(ruta)
    try:
        if not 1 <= numero <= len(documento):
            raise ValueError(f'el PDF tiene {len(documento)} páginas')
        pagina = documento[numero - 1]
        imagen = _imagen_unica(pagina)
        return _a_bgr(imagen.get_bitmap() if imagen else _dibujar(pagina))
    finally:
        documento.close()


def _abrir(ruta):
    """Documento de pypdfium2. Si no se puede, OSError con el motivo en palabras."""
    try:
        import pypdfium2
    except ImportError:
        raise OSError(
            'para leer PDF falta la biblioteca pypdfium2. Cierra y vuelve a abrir con '
            'el lanzador, con internet, para que la instale.'
        ) from None
    try:
        return pypdfium2.PdfDocument(str(ruta))
    except pypdfium2.PdfiumError as exc:
        con_clave = exc.err_code == pypdfium2.raw.FPDF_ERR_PASSWORD
        raise OSError(
            'el PDF tiene clave' if con_clave else 'no se pudo leer como PDF'
        ) from exc


def _imagen_unica(pagina):
    """La imagen de una página escaneada, o None si la página tiene otra cosa.

    Debe ser lo único que se ve (el texto invisible de un OCR no cuenta), estar
    derecha, sin transparencia y en grises o RGB de 8 bits: así sus píxeles son los
    que se verían en pantalla.
    """
    import pypdfium2.raw as raw

    visibles = []
    for objeto in pagina.get_objects(max_depth=1):
        if objeto.type == raw.FPDF_PAGEOBJ_TEXT:
            modo = raw.FPDFTextObj_GetTextRenderMode(objeto)
            if modo == raw.FPDF_TEXTRENDERMODE_INVISIBLE:
                continue
        visibles.append(objeto)
    if len(visibles) != 1 or visibles[0].type != raw.FPDF_PAGEOBJ_IMAGE:
        return None
    imagen = visibles[0]
    a, b, c, d, _, _ = imagen.get_matrix().get()
    derecha = a > 0 and d > 0 and abs(b) < 1e-3 * a and abs(c) < 1e-3 * d
    if pagina.get_rotation() or not derecha or raw.FPDFPageObj_HasTransparency(imagen):
        return None
    datos = imagen.get_metadata()
    simple = datos.colorspace in (
        raw.FPDF_COLORSPACE_DEVICEGRAY,
        raw.FPDF_COLORSPACE_DEVICERGB,
        raw.FPDF_COLORSPACE_CALGRAY,
        raw.FPDF_COLORSPACE_CALRGB,
        raw.FPDF_COLORSPACE_ICCBASED,
    )
    return imagen if simple and datos.bits_per_pixel in (8, 24) else None


def _dibujar(pagina):
    """Dibuja la página completa, sobre blanco, con la resolución de su imagen principal
    (la que ocupa al menos un cuarto de la página)."""
    import pypdfium2.raw as raw

    ancho, alto = pagina.get_size()
    escala = PPP / PUNTOS_POR_PULGADA
    for imagen in pagina.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=1):
        x1, y1, x2, y2 = imagen.get_bounds()
        if (x2 - x1) * (y2 - y1) >= ancho * alto / 4:
            escala = max(imagen.get_px_size()) / max(x2 - x1, y2 - y1)
            break
    return pagina.render(scale=min(escala, LADO_MAXIMO / max(ancho, alto)))


def _a_bgr(mapa) -> np.ndarray:
    """Copia de un mapa de bits de PDFium como imagen BGR."""
    pixeles = mapa.to_numpy()
    if pixeles.ndim == 2 or pixeles.shape[2] == 1:
        return cv2.cvtColor(pixeles, cv2.COLOR_GRAY2BGR)
    return pixeles[:, :, :3].copy()
