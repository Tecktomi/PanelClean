"""Muestreo estratificado de las regiones reales que se etiquetan a mano.

En OpenMantra casi todo el texto está en globos blancos: en una muestra al azar
de 107 regiones, 103 tenían fondo liso. Mostrarlas al azar obligaría a etiquetar
casi todo el dataset para juntar 15 regiones con trama o dibujo.

Por eso las regiones se reparten en 4 estratos según una medida simple de cuán
cargado está el fondo dentro de la caja (la fracción de píxeles grises medios,
ni tinta ni papel), y al etiquetar se muestran más a menudo las del estrato más
cargado. Dentro de cada estrato el orden es al azar.

Como unas regiones se muestran más que otras, la evaluación pondera cada región
etiquetada por N_h / n_h: regiones del estrato h en todo el dataset dividido por
regiones etiquetadas de ese estrato. Así la estimación vale para todo OpenMantra
y no solo para las regiones mostradas.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

from panelclean import vision as V

CARPETA = Path(__file__).resolve().parent
DATASET = CARPETA.parent / 'datasets' / 'open-mantra'
ESTRATOS = CARPETA / 'estratos.csv'
SEMILLA = 2026
# Percentiles de la medida que separan los estratos: el estrato 1 es el 5 % más
# cargado, el 2 va del percentil 80 al 95, el 3 del 50 al 80 y el 4 es el resto.
CORTES = (95, 80, 50)
# Probabilidad de que la siguiente región mostrada salga de cada estrato.
PROPORCION = {1: 0.55, 2: 0.25, 3: 0.10, 4: 0.10}

Clave = tuple[str, int, int]  # (libro, página, número de caja de texto)


def medida(gris_caja: np.ndarray) -> float:
    """Fracción de píxeles grises medios (entre 60 y 200) dentro de la caja.

    En un globo blanco casi todo es papel (claro) o tinta (oscura); la trama, los
    degradados y el dibujo aportan muchos grises intermedios.
    """
    return float(((gris_caja > 60) & (gris_caja < 200)).mean())


def calcular_estratos() -> list[dict]:
    """Mide todas las cajas de texto de OpenMantra y les asigna un estrato."""
    datos = json.loads((DATASET / 'annotation.json').read_text(encoding='utf-8'))
    filas = []
    for libro in datos:
        for pagina in libro['pages']:
            imagen = V.bgr_from_file(str(DATASET / pagina['image_paths']['ja']))
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            for k, t in enumerate(pagina['text']):
                caja = gris[t['y'] : t['y'] + t['h'], t['x'] : t['x'] + t['w']]
                filas.append(
                    {
                        'libro': libro['book_title'],
                        'pagina': pagina['page_index'],
                        'texto': k,
                        'grises': round(medida(caja), 4),
                    }
                )
    valores = np.array([f['grises'] for f in filas])
    limites = [float(np.percentile(valores, c)) for c in CORTES]
    for f in filas:
        f['estrato'] = 1 + sum(f['grises'] < limite for limite in limites)
    return filas


def cargar_estratos() -> dict[Clave, int]:
    """Estrato de cada región. La primera vez lo calcula y lo guarda en estratos.csv."""
    if not ESTRATOS.exists():
        filas = calcular_estratos()
        with open(ESTRATOS, 'w', newline='', encoding='utf-8') as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
            escritor.writeheader()
            escritor.writerows(filas)
    with open(ESTRATOS, encoding='utf-8') as f:
        return {
            (e['libro'], int(e['pagina']), int(e['texto'])): int(e['estrato'])
            for e in csv.DictReader(f)
        }


def ordenar(claves: list[Clave]) -> list[Clave]:
    """Orden de etiquetado: cada región sale de un estrato elegido según PROPORCION.

    Es el mismo para todas las personas (semilla fija), así pueden comparar sus
    etiquetas. Cuando un estrato se agota, se sigue con los demás.
    """
    estratos = cargar_estratos()
    rng = random.Random(SEMILLA)
    grupos = {h: sorted(c for c in claves if estratos[c] == h) for h in PROPORCION}
    for g in grupos.values():
        rng.shuffle(g)
    orden = []
    while any(grupos.values()):
        activos = [h for h in PROPORCION if grupos[h]]
        h = rng.choices(activos, weights=[PROPORCION[a] for a in activos])[0]
        orden.append(grupos[h].pop())
    return orden


def pesos(
    etiquetadas: list[Clave],
) -> tuple[dict[Clave, float], dict[int, tuple[int, int]]]:
    """Peso N_h / n_h de cada región etiquetada y, por estrato, (n_h, N_h)."""
    estratos = cargar_estratos()
    total = {h: 0 for h in PROPORCION}
    for h in estratos.values():
        total[h] += 1
    muestra = {h: 0 for h in PROPORCION}
    for c in etiquetadas:
        muestra[estratos[c]] += 1
    salida = {c: total[estratos[c]] / muestra[estratos[c]] for c in etiquetadas}
    return salida, {h: (muestra[h], total[h]) for h in PROPORCION}
