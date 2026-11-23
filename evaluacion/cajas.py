"""Cajas de texto marcadas a mano en páginas propias (las guarda marcar.py).

Se guardan en dos archivos:
- cajas_texto.json: por serie y por página, los rectángulos marcados. No incluye
  imágenes ni rutas del equipo, así que se puede compartir.
- carpetas_locales.json: en qué carpeta de este equipo está cada serie. No se sube
  al repositorio (está en .gitignore).

Formato de cajas_texto.json:
    {"mi_serie": {"reservada": false, "paginas": {"001.jpg": {
        "ancho": 1600, "alto": 2300, "completa": true, "etiquetador": "Ana",
        "cajas": [{"x": 10, "y": 20, "w": 80, "h": 120,
                   "tipo": "texto", "origen": "mano"}]}}}}

tipo: 'texto' o 'ignorar' (onomatopeyas y otro texto que es parte del dibujo: no
cuenta ni como texto ni como «no texto»). origen: 'mano' o 'propuesta' (la puso el
detector y nadie la cambió). Solo se usan las páginas completas: en ellas, todo lo
que no está marcado se aprende como «no texto».
Una serie reservada nunca se usa para entrenar ni para elegir nada: solo para medir.
"""

from __future__ import annotations

import json
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
CAJAS = CARPETA / 'cajas_texto.json'
LOCALES = CARPETA / 'carpetas_locales.json'
EXTENSIONES = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

Caja = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def _leer(ruta: Path) -> dict:
    """Contenido de un JSON (vacío si no existe)."""
    return json.loads(ruta.read_text(encoding='utf8')) if ruta.exists() else {}


def _escribir(ruta: Path, datos: dict) -> None:
    """Guarda un JSON, primero en un archivo temporal: un corte no pierde lo anterior."""
    temporal = ruta.with_suffix('.tmp')
    temporal.write_text(
        json.dumps(datos, ensure_ascii=False, indent=1, sort_keys=True) + '\n',
        encoding='utf8',
    )
    temporal.replace(ruta)


def leer() -> dict:
    """Todas las series marcadas."""
    return _leer(CAJAS)


def guardar(series: dict) -> None:
    """Guarda todas las series marcadas."""
    _escribir(CAJAS, series)


def paginas_de(carpeta: Path) -> list[Path]:
    """Imágenes de una carpeta, en orden."""
    return sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)


def registrar_carpeta(carpeta: Path) -> str:
    """Nombre de la serie que corresponde a la carpeta; recuerda dónde está.

    Es el nombre de la carpeta; si ya lo usa otra carpeta, se le agrega un número.
    """
    carpeta = carpeta.resolve()
    locales = _leer(LOCALES)
    for serie, ruta in locales.items():
        if Path(ruta) == carpeta:
            return serie
    base = serie = carpeta.name or 'serie'
    numero = 2
    while serie in locales and Path(locales[serie]).exists():
        serie = f'{base} ({numero})'
        numero += 1
    locales[serie] = str(carpeta)
    _escribir(LOCALES, locales)
    return serie


def carpeta_de(serie: str) -> Path | None:
    """Carpeta de este equipo con las imágenes de la serie, o None."""
    ruta = _leer(LOCALES).get(serie)
    return Path(ruta) if ruta and Path(ruta).exists() else None


def completas(serie: dict) -> dict[str, dict]:
    """Páginas completas de una serie."""
    return {n: p for n, p in serie['paginas'].items() if p.get('completa')}


def rectangulos(pagina: dict, tipo: str) -> list[Caja]:
    """Cajas (x1, y1, x2, y2) de una página con el tipo dado."""
    return [
        (c['x'], c['y'], c['x'] + c['w'], c['y'] + c['h'])
        for c in pagina['cajas']
        if c['tipo'] == tipo
    ]


def area(c: Caja) -> float:
    """Área de un rectángulo (x1, y1, x2, y2)."""
    return max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])


def centro_en(d: Caja, t: Caja) -> bool:
    """Indica si el centro de d cae dentro de t."""
    cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
    return t[0] <= cx < t[2] and t[1] <= cy < t[3]


def acierta(d: Caja, t: Caja) -> bool:
    """Indica si la región d corresponde al texto t: su centro cae en t y no mide
    más de 3 veces su área (una región de media hoja no es un acierto aunque su
    centro caiga, por casualidad, en un texto)."""
    return centro_en(d, t) and area(d) <= 3 * area(t)


def texto_del_candidato(d: Caja, textos: list[Caja]) -> int:
    """Índice del texto al que corresponde el candidato d, o -1 si no es texto.

    Además de `acierta`, vale un candidato que cubre varios textos juntos (globos
    unidos): al menos el 80 % de cada uno, sin medir más de 3 veces lo que miden.
    """
    for k, t in enumerate(textos):
        if acierta(d, t):
            return k
    cubiertos = []
    for k, t in enumerate(textos):
        comun = area(
            (max(d[0], t[0]), max(d[1], t[1]), min(d[2], t[2]), min(d[3], t[3]))
        )
        if comun >= 0.8 * area(t) > 0:
            cubiertos.append(k)
    if cubiertos and area(d) <= 3 * sum(area(textos[k]) for k in cubiertos):
        return cubiertos[0]
    return -1
