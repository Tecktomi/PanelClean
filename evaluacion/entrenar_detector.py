"""Entrena el detector de texto de PanelClean (tarea T1.2.6).

El detector propone como candidatos los bloques de texto de la página entera y los
describe con las medidas de `vision.RASGOS_TEXTO`. Este script arma los datos, mide
el detector con validación cruzada por obra y guarda el random forest final en
`panelclean/modelo_texto.yml`.

Datos:
- OpenMantra (`candidatos.csv`, se genera solo). Cada página de desarrollo se usa
  en tres versiones: la original, la traspuesta (columnas que pasan a renglones) y
  una con los globos traducidos en renglones (horizontales.py). La obra reservada
  (`tencho_isoro`) nunca se usa para entrenar.
- Páginas propias marcadas con `python -m evaluacion.marcar` (cajas_texto.json). Las
  series reservadas no se usan para entrenar: solo se informa cómo le va en ellas.
  Las medidas de sus candidatos quedan en `candidatos_propios.csv`, así se puede
  reentrenar en un equipo que no tiene las imágenes.

Un candidato es texto si corresponde a una caja (cajas.texto_del_candidato).

Uso, desde la raíz del repositorio:
    .venv\\Scripts\\python -m evaluacion.entrenar_detector
    .venv\\Scripts\\python -m evaluacion.entrenar_detector --recalcular
Con --recalcular se vuelven a medir las páginas de OpenMantra (se hace solo si
cambió vision.VERSION_RASGOS).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from panelclean import vision as V

from . import cajas as K
from . import horizontales

CARPETA = Path(__file__).resolve().parent
DATASET = CARPETA.parent / 'datasets' / 'open-mantra'
CANDIDATOS = CARPETA / 'candidatos.csv'
PROPIOS = CARPETA / 'candidatos_propios.csv'
LIBRO_PRUEBA = 'tencho_isoro'
# 30 árboles de profundidad 7 rinden igual que bosques más grandes en la validación
# cruzada, y el modelo pesa menos (≈ 0,6 MB).
ARBOLES, PROFUNDIDAD, MINIMO_HOJA = 30, 7, 10
# Las páginas propias se repiten hasta pesar esta fracción de OpenMantra: son pocas,
# pero son del estilo que se quiere limpiar.
PESO_PROPIAS, REPETIR_MAXIMO = 0.4, 10
CAMPOS = [
    'version',
    'grupo',
    'pagina',
    'tipo',  # original, traspuesta, horizontal o propia
    'x',
    'y',
    'w',
    'h',
    'es_texto',
    'texto',
    'limpiable',
    'px_texto',
    'px_dibujo',
    *V.RASGOS_TEXTO,
]


# ---------------------------------------------------------------------------
# Candidatos
# ---------------------------------------------------------------------------


def cargar_openmantra() -> list[dict]:
    """Anotaciones de OpenMantra."""
    return json.loads((DATASET / 'annotation.json').read_text(encoding='utf-8'))


def _cajas(pagina: dict) -> list[tuple]:
    """Cajas de texto anotadas (x1, y1, x2, y2) de una página de OpenMantra."""
    return [(t['x'], t['y'], t['x'] + t['w'], t['y'] + t['h']) for t in pagina['text']]


def medir_pagina(
    imagen: np.ndarray,
    textos: list[tuple],
    ignorar: list[tuple],
    comun: dict,
    con_limpieza: bool = False,
) -> list[dict]:
    """Una fila por candidato de la página, con sus medidas y si es texto.

    Se omiten los candidatos que caen en una caja a ignorar. Con con_limpieza se
    agrega qué borraría el modo automático: si la región se limpiaría según su fondo
    (limpiable) y cuántos píxeles de su máscara caen dentro y fuera de los textos.
    """
    en_textos = np.zeros(imagen.shape[:2], bool)
    for x1, y1, x2, y2 in textos:
        en_textos[y1:y2, x1:x2] = True
    filas = []
    for r, medidas in zip(*V.candidatos_de_texto(imagen)):
        d = (r.x, r.y, r.x + r.w, r.y + r.h)
        k = K.texto_del_candidato(d, textos)
        if k < 0 and any(K.centro_en(d, i) for i in ignorar):
            continue
        fila = {
            'version': V.VERSION_RASGOS,
            **comun,
            'x': r.x,
            'y': r.y,
            'w': r.w,
            'h': r.h,
            'es_texto': int(k >= 0),
            'texto': k,
            'limpiable': '',
            'px_texto': '',
            'px_dibujo': '',
            **{n: round(float(v), 6) for n, v in zip(V.RASGOS_TEXTO, medidas)},
        }
        if con_limpieza:
            mascara = V.generate_mask(imagen, r, V.Options()) > 0
            zona = en_textos[r.y : r.y + r.h, r.x : r.x + r.w]
            fila['limpiable'] = int(r.suggested_method != 'Revisión manual')
            fila['px_texto'] = int((mascara & zona).sum())
            fila['px_dibujo'] = int((mascara & ~zona).sum())
        filas.append(fila)
    return filas


def medir_openmantra(datos: list[dict], avance=print) -> list[dict]:
    """Candidatos de todas las páginas de OpenMantra. La obra reservada se mide solo
    en su versión original: no se entrena con ella."""
    filas = []
    inicio = time.time()
    for libro in datos:
        nombre = libro['book_title']
        for pagina in libro['pages']:
            base = V.bgr_from_file(str(DATASET / pagina['image_paths']['ja']))
            textos = _cajas(pagina)
            comun = {'grupo': nombre, 'pagina': pagina['page_index']}
            filas += medir_pagina(
                base, textos, [], {**comun, 'tipo': 'original'}, con_limpieza=True
            )
            if nombre == LIBRO_PRUEBA:
                continue
            filas += medir_pagina(
                np.ascontiguousarray(base.transpose(1, 0, 2)),
                [(b, a, d, c) for a, b, c, d in textos],
                [],
                {**comun, 'tipo': 'traspuesta'},
            )
            rng = horizontales.azar(nombre, pagina['page_index'])
            traducida, cajas = horizontales.traducir(base, pagina['text'], rng)
            filas += medir_pagina(traducida, cajas, [], {**comun, 'tipo': 'horizontal'})
        avance(f'  {nombre}: {len(filas)} candidatos ({time.time() - inicio:.0f} s)')
    return filas


def medir_propias(avance=print) -> list[dict]:
    """Candidatos de las páginas marcadas como listas.

    Se miden las series cuyas imágenes están en este equipo; de las demás se usan
    las medidas guardadas, si son de la misma versión de los candidatos.
    """
    guardadas = [f for f in leer_csv(PROPIOS) if f['version'] == str(V.VERSION_RASGOS)]
    filas = []
    for grupo, serie in sorted(K.leer().items()):
        paginas = K.completas(serie)
        carpeta = K.carpeta_de(grupo)
        if carpeta is None:
            previas = [
                f for f in guardadas if f['grupo'] == grupo and f['pagina'] in paginas
            ]
            if paginas and not previas:
                avance(f'  {grupo}: sin sus imágenes ni medidas al día; se omite.')
            filas += previas
            continue
        n = 0
        for archivo, pagina in sorted(paginas.items()):
            ruta = carpeta / archivo
            if not ruta.exists():
                continue
            nuevas = medir_pagina(
                V.bgr_from_file(str(ruta)),
                K.rectangulos(pagina, 'texto'),
                K.rectangulos(pagina, 'ignorar'),
                {'grupo': grupo, 'pagina': archivo, 'tipo': 'propia'},
            )
            filas += [{k: str(v) for k, v in f.items()} for f in nuevas]
            n += len(nuevas)
        avance(f'  {grupo}: {len(paginas)} páginas listas, {n} candidatos')
    if filas:
        escribir_csv(PROPIOS, CAMPOS, filas)
    return filas


def leer_csv(ruta: Path) -> list[dict]:
    """Filas de un CSV (lista vacía si no existe)."""
    if not ruta.exists():
        return []
    with open(ruta, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def escribir_csv(ruta: Path, campos: list[str], filas: list[dict]) -> None:
    """Escribe un CSV completo."""
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def candidatos_openmantra(recalcular: bool = False, avance=print) -> list[dict]:
    """Candidatos de OpenMantra, desde candidatos.csv o medidos de nuevo."""
    filas = [] if recalcular else leer_csv(CANDIDATOS)
    vigente = (
        filas
        and list(filas[0]) == CAMPOS
        and filas[0]['version'] == str(V.VERSION_RASGOS)
    )
    if not vigente:
        avance('Midiendo los candidatos de OpenMantra (unos 5 minutos)…')
        escribir_csv(CANDIDATOS, CAMPOS, medir_openmantra(cargar_openmantra(), avance))
        filas = leer_csv(CANDIDATOS)
    return filas


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def matriz(filas: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(medidas, etiqueta 0/1) de las filas."""
    X = np.array([[float(f[k]) for k in V.RASGOS_TEXTO] for f in filas], np.float32)
    y = np.array([int(f['es_texto']) for f in filas], np.int32)
    return X.reshape(-1, len(V.RASGOS_TEXTO)), y


def entrenar(X: np.ndarray, y: np.ndarray):
    """Random forest de OpenCV con los parámetros del detector."""
    modelo = cv2.ml.RTrees_create()
    modelo.setMaxDepth(PROFUNDIDAD)
    modelo.setMinSampleCount(MINIMO_HOJA)
    modelo.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, ARBOLES, 0))
    modelo.setCalculateVarImportance(True)
    modelo.train(X, cv2.ml.ROW_SAMPLE, y)
    return modelo


def entrenar_con(filas: list[dict], propias: list[dict]):
    """Entrena con filas de OpenMantra y de páginas propias, repitiendo estas
    últimas hasta que pesen PESO_PROPIAS de las primeras."""
    X, y = matriz(filas)
    if propias:
        Xp, yp = matriz(propias)
        veces = min(
            REPETIR_MAXIMO, max(1, round(PESO_PROPIAS * len(filas) / len(propias)))
        )
        X = np.vstack([X] + [Xp] * veces)
        y = np.concatenate([y] + [yp] * veces)
    return entrenar(X, y)


def guardar(modelo, ruta: Path = V.MODELO_TEXTO) -> None:
    """Guarda el modelo como YAML (escrito por Python: funciona con tildes en la ruta)."""
    fs = cv2.FileStorage('.yml', cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
    modelo.write(fs)
    Path(ruta).write_text(fs.releaseAndGetString(), encoding='utf8')
    V._modelos.pop(Path(ruta), None)


def validacion_cruzada(
    filas: list[dict], propias: list[dict] = (), reservadas: set[str] = frozenset()
) -> tuple[np.ndarray, np.ndarray, object]:
    """Confianza de cada fila medida sin ver su grupo, y el modelo final.

    Cada obra de desarrollo, y cada serie propia no reservada, se predice con un
    modelo entrenado sin ella. La obra de prueba y las series reservadas, con el
    modelo final, que no las usa. Devuelve (confianzas de OpenMantra, confianzas de
    las propias, modelo final).
    """
    propias = list(propias)
    desarrollo = [f for f in filas if f['grupo'] != LIBRO_PRUEBA]
    entrenables = [f for f in propias if f['grupo'] not in reservadas]
    final = entrenar_con(desarrollo, entrenables)

    def predecir(conjunto, sin_ver):
        """Confianza de las filas del conjunto, con un modelo que no vio cada grupo."""
        p = np.zeros(len(conjunto))
        for g in dict.fromkeys(f['grupo'] for f in conjunto):
            sel = [i for i, f in enumerate(conjunto) if f['grupo'] == g]
            modelo = final
            if g in sin_ver:
                modelo = entrenar_con(
                    [f for f in desarrollo if f['grupo'] != g],
                    [f for f in entrenables if f['grupo'] != g],
                )
            X, _ = matriz([conjunto[i] for i in sel])
            p[sel] = V.probabilidad_de_texto(modelo, X)
        return p

    p = predecir(filas, {f['grupo'] for f in desarrollo})
    pp = predecir(propias, {f['grupo'] for f in entrenables})
    return p, pp, final


def validacion_por_paginas(
    filas: list[dict], propias: list[dict], grupo: str, partes: int = 5
) -> np.ndarray | None:
    """Confianza de cada candidato de una serie propia, medida con un modelo que sí
    vio otras páginas de esa serie, pero no la suya (validación cruzada por páginas).

    Responde cuánto se gana enseñándole páginas del mismo estilo. None si la serie
    tiene menos de 10 páginas.
    """
    de_serie = [f for f in propias if f['grupo'] == grupo]
    paginas = sorted({f['pagina'] for f in de_serie})
    if len(paginas) < 10:
        return None
    desarrollo = [f for f in filas if f['grupo'] != LIBRO_PRUEBA]
    orden = np.random.default_rng(0).permutation(len(paginas))
    p = np.zeros(len(de_serie))
    for k in range(partes):
        fuera = {paginas[i] for i in orden[k::partes]}
        sel = [i for i, f in enumerate(de_serie) if f['pagina'] in fuera]
        modelo = entrenar_con(
            desarrollo, [f for f in de_serie if f['pagina'] not in fuera]
        )
        X, _ = matriz([de_serie[i] for i in sel])
        p[sel] = V.probabilidad_de_texto(modelo, X)
    return p


# ---------------------------------------------------------------------------
# Medición
# ---------------------------------------------------------------------------


def textos_por_grupo(datos: list[dict]) -> dict[str, int]:
    """Cantidad de cajas de texto anotadas en cada obra de OpenMantra."""
    return {
        libro['book_title']: sum(len(p['text']) for p in libro['pages'])
        for libro in datos
    }


def textos_propios() -> dict[str, int]:
    """Cantidad de cajas de texto marcadas en las páginas listas de cada serie."""
    return {
        grupo: sum(len(K.rectangulos(p, 'texto')) for p in K.completas(serie).values())
        for grupo, serie in K.leer().items()
    }


def medir_deteccion(
    filas: list[dict],
    p: np.ndarray,
    umbral: float,
    textos: int,
    tipo: str = 'original',
) -> dict:
    """Precisión, recall y F1 aceptando confianza >= umbral, en las páginas del tipo
    dado, y el techo: el recall si se aceptaran todos los candidatos. Con px_texto y
    px_dibujo (solo en las originales), lo que borraría el modo automático."""
    propias = [(f, q) for f, q in zip(filas, p) if f['tipo'] == tipo]
    aceptados = [f for f, q in propias if q >= umbral]

    def hallados(conjunto):
        """Textos distintos a los que corresponde algún candidato del conjunto."""
        return len(
            {
                (f['grupo'], f['pagina'], f['texto'])
                for f in conjunto
                if f['es_texto'] == '1'
            }
        )

    precision = sum(f['es_texto'] == '1' for f in aceptados) / max(1, len(aceptados))
    recall = hallados(aceptados) / max(1, textos)
    limpias = [f for f in aceptados if f['limpiable'] == '1']
    return {
        'detecciones': len(aceptados),
        'textos': textos,
        'precision': precision,
        'recall': recall,
        'techo': hallados([f for f, _ in propias]) / max(1, textos),
        'f1': 2 * precision * recall / max(1e-9, precision + recall),
        'px_texto': sum(int(f['px_texto']) for f in limpias),
        'px_dibujo': sum(int(f['px_dibujo']) for f in limpias),
    }


def _linea(nombre: str, m: dict) -> str:
    """Resumen de una medición en una línea."""
    return (
        f'  {nombre}: {m["textos"]} textos · candidatos {m["techo"]:.0%} · '
        f'P {m["precision"]:.2f} · R {m["recall"]:.2f} · F1 {m["f1"]:.2f}'
    )


def main() -> None:
    """Punto de entrada de la línea de comandos."""
    main_con(None)


def main_con(argv: list[str] | None) -> None:
    """Arma los datos, informa la validación cruzada y guarda el modelo final.

    argv son los argumentos (None = los de la línea de comandos); marcar.py la llama
    con [] para entrenar al cerrar su ventana.
    """
    parser = argparse.ArgumentParser(description='Entrenar el detector de texto.')
    parser.add_argument(
        '--recalcular', action='store_true', help='volver a medir OpenMantra'
    )
    args = parser.parse_args(argv)
    filas = candidatos_openmantra(args.recalcular)
    print('Páginas propias:')
    propias = medir_propias()
    if not propias:
        print('  todavía no hay (se marcan con python -m evaluacion.marcar)')
    reservadas = {g for g, s in K.leer().items() if s['reservada']}
    p, pp, modelo = validacion_cruzada(filas, propias, reservadas)

    totales = textos_por_grupo(cargar_openmantra())
    dev = np.array([f['grupo'] != LIBRO_PRUEBA for f in filas])
    print(f'\nOpenMantra, confianza ≥ {V.CONFIANZA_LIMPIAR} (cada obra sin verla):')
    for nombre, sel in (
        ('desarrollo', dev),
        (f'{LIBRO_PRUEBA} (reservada, ya consultada)', ~dev),
    ):
        sub = [f for f, s in zip(filas, sel) if s]
        n = sum(totales[g] for g in {f['grupo'] for f in sub})
        print(_linea(nombre, medir_deteccion(sub, p[sel], V.CONFIANZA_LIMPIAR, n)))
    propios = textos_propios()
    if propias:
        print(f'\nSeries propias, confianza ≥ {V.CONFIANZA_LIMPIAR}:')
    for grupo in dict.fromkeys(f['grupo'] for f in propias):
        sel = np.array([f['grupo'] == grupo for f in propias])
        sub = [f for f, s in zip(propias, sel) if s]
        m = medir_deteccion(sub, pp[sel], V.CONFIANZA_LIMPIAR, propios[grupo], 'propia')
        marca = 'reservada' if grupo in reservadas else 'entrenando sin ella'
        print(_linea(f'{grupo} ({marca})', m))
        if grupo not in reservadas:
            q = validacion_por_paginas(filas, propias, grupo)
            if q is not None:
                m = medir_deteccion(
                    sub, q, V.CONFIANZA_LIMPIAR, propios[grupo], 'propia'
                )
                print(_linea(f'{grupo} (entrenando con otras de sus páginas)', m))
    if propias:
        print(
            '\n«candidatos» es el techo: si es bajo, el texto no se propone (está '
            'sobre dibujo o en globos oscuros) y entrenar no lo arregla.'
        )
    guardar(modelo)
    importancia = modelo.getVarImportance().ravel()
    orden = np.argsort(-importancia)[:6]
    print(
        '\nMedidas más usadas: '
        + ', '.join(f'{V.RASGOS_TEXTO[i]} {importancia[i]:.2f}' for i in orden)
    )
    print(f'Modelo guardado en {V.MODELO_TEXTO}.')


if __name__ == '__main__':
    main()
