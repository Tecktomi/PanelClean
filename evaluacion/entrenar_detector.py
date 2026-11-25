"""Entrena el detector de texto de PanelClean (tarea T1.2.6).

El detector propone como candidatos los bloques de texto de la página entera y los
describe con las medidas de `vision.RASGOS_TEXTO`. Este script arma los datos, mide
el detector con validación cruzada por obra y guarda el random forest final en
`panelclean/modelo_texto.yml`.

Datos:
- OpenMantra (`candidatos.csv.gz`, se genera solo). Cada página de desarrollo se usa
  en cuatro versiones: la original, la traspuesta (columnas que pasan a renglones),
  una con los globos traducidos en renglones (horizontales.py) y esa misma como un
  escaneo chico y comprimido. La obra reservada (`tencho_isoro`) nunca se usa para
  entrenar.
- Páginas propias marcadas con `python -m evaluacion.marcar` (cajas_texto.json). Las
  series reservadas no se usan para entrenar: solo se informa cómo le va en ellas.
  Las medidas de sus candidatos quedan en `candidatos_propios.csv`, así se puede
  reentrenar en un equipo que no tiene las imágenes.

Los candidatos se buscan con varios tamaños de letra, así que hay bloques a medias.
Para entrenar, un candidato es texto si contiene textos enteros (LETRAS_ENTERAS) sin
medir más de 3 veces lo que miden; un trozo de un texto no es ejemplo de nada. Para
medir, los candidatos se eligen como en la aplicación (vision.sin_solapes) y acierta
el que corresponde a una caja (cajas.texto_del_candidato), como siempre.

Uso, desde la raíz del repositorio:
    .venv\\Scripts\\python -m evaluacion.entrenar_detector
    .venv\\Scripts\\python -m evaluacion.entrenar_detector --recalcular
Con --recalcular se vuelven a medir las páginas de OpenMantra (se hace solo si
cambió vision.VERSION_RASGOS).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
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
CANDIDATOS = CARPETA / 'candidatos.csv.gz'
PROPIOS = CARPETA / 'candidatos_propios.csv'
LIBRO_PRUEBA = 'tencho_isoro'
# 30 árboles de profundidad 7 rinden igual que bosques más grandes en la validación
# cruzada, y el modelo pesa menos (≈ 0,6 MB).
ARBOLES, PROFUNDIDAD, MINIMO_HOJA = 30, 7, 10
SEMILLA = 2026  # el bosque se entrena con muestras al azar: así sale siempre igual
# Parte de las letras de un texto que debe contener un bloque para tenerlo entero.
LETRAS_ENTERAS = 0.85
# Las páginas propias se repiten hasta pesar esta fracción de OpenMantra: son pocas,
# pero son del estilo que se quiere limpiar.
PESO_PROPIAS, REPETIR_MAXIMO = 0.4, 10
CAMPOS = [
    'version',
    'grupo',
    'pagina',
    'tipo',  # original, traspuesta, horizontal, horizontal_baja o propia
    'x',
    'y',
    'w',
    'h',
    'es_texto',
    'texto',
    'enteros',  # textos que contiene enteros, separados por «;»
    'cubre',  # parte de las letras de cada texto que contiene: «texto:parte;…»
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


def _letras(pagina: V.Pagina, caja: tuple) -> tuple[tuple, np.ndarray]:
    """Letras de un texto anotado: su caja y la máscara de su tinta, en píxeles de
    trabajo. No cuenta lo que toca el borde de la caja (el contorno del globo), salvo
    que sea casi todo."""
    e = pagina.escala
    alto, ancho = pagina.limpio.shape
    x1, y1 = max(0, int(caja[0] * e)), max(0, int(caja[1] * e))
    x2 = min(ancho, int(np.ceil(caja[2] * e)))
    y2 = min(alto, int(np.ceil(caja[3] * e)))
    tinta = (pagina.limpio[y1:y2, x1:x2] <= V.TINTA_PIEZA).astype(np.uint8)
    _, piezas, st, _ = cv2.connectedComponentsWithStats(tinta, 8)
    toca = (st[:, 0] == 0) | (st[:, 1] == 0)
    toca |= (st[:, 0] + st[:, 2] >= x2 - x1) | (st[:, 1] + st[:, 3] >= y2 - y1)
    toca[0] = True
    libre = ~toca[piezas]
    if libre.sum() < 0.5 * tinta.sum():
        libre = tinta > 0
    return (x1, y1, x2, y2), libre


def _parte_cubierta(caja: tuple, letras: tuple[tuple, np.ndarray]) -> float:
    """Parte de las letras de un texto que queda dentro de la caja (de trabajo)."""
    (x1, y1, x2, y2), tinta = letras
    xa, ya = max(caja[0], x1), max(caja[1], y1)
    xb, yb = min(caja[2], x2), min(caja[3], y2)
    if xb <= xa or yb <= ya:
        return 0.0
    dentro = tinta[ya - y1 : yb - y1, xa - x1 : xb - x1].sum()
    return float(dentro / max(1, tinta.sum()))


def medir_pagina(
    imagen: np.ndarray,
    textos: list[tuple],
    ignorar: list[tuple],
    comun: dict,
    con_limpieza: bool = False,
) -> list[dict]:
    """Una fila por candidato de la página, con sus medidas y si es texto.

    Se omiten los candidatos que caen en una caja a ignorar. De cada uno se guarda
    también qué parte de las letras de cada texto contiene. Con con_limpieza se
    agrega qué borraría el modo automático: si la región se limpiaría según su fondo
    (limpiable) y cuántos píxeles de su máscara caen dentro y fuera de los textos.
    """
    en_textos = np.zeros(imagen.shape[:2], bool)
    for x1, y1, x2, y2 in textos:
        en_textos[y1:y2, x1:x2] = True
    filas = []
    preparada = V.preparar_pagina(imagen)
    letras = [_letras(preparada, t) for t in textos]
    for r, medidas in zip(*V.candidatos_de_texto(imagen, preparada)):
        d = (r.x, r.y, r.x + r.w, r.y + r.h)
        k = K.texto_del_candidato(d, textos)
        if k < 0 and any(K.centro_en(d, i) for i in ignorar):
            continue
        trabajo = tuple(int(round(v * preparada.escala)) for v in d)
        cubre = {j: _parte_cubierta(trabajo, q) for j, q in enumerate(letras)}
        enteros = [j for j, v in cubre.items() if v >= LETRAS_ENTERAS]
        if K.area(d) > 3 * sum(K.area(textos[j]) for j in enteros):
            enteros = []
        fila = {
            'version': V.VERSION_RASGOS,
            **comun,
            'x': r.x,
            'y': r.y,
            'w': r.w,
            'h': r.h,
            'es_texto': int(k >= 0),
            'texto': k,
            'enteros': ';'.join(map(str, enteros)),
            'cubre': ';'.join(f'{j}:{v:.2f}' for j, v in cubre.items() if v >= 0.01),
            'limpiable': '',
            'px_texto': '',
            'px_dibujo': '',
            **{n: round(float(v), 4) for n, v in zip(V.RASGOS_TEXTO, medidas)},
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
            baja, cajas = horizontales.degradar(traducida, cajas, rng)
            filas += medir_pagina(baja, cajas, [], {**comun, 'tipo': 'horizontal_baja'})
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
            ruta = K.ruta_de(carpeta, archivo)
            if ruta is None:
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
    """Filas de un CSV, comprimido (.gz) o no (lista vacía si no existe)."""
    if not ruta.exists():
        return []
    abrir = gzip.open if ruta.suffix == '.gz' else open
    with abrir(ruta, 'rt', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def escribir_csv(ruta: Path, campos: list[str], filas: list[dict]) -> None:
    """Escribe un CSV completo, comprimido si la ruta termina en .gz."""
    texto = io.StringIO(newline='')
    escritor = csv.DictWriter(texto, fieldnames=campos)
    escritor.writeheader()
    escritor.writerows(filas)
    datos = texto.getvalue().encode('utf-8')
    # Sin la fecha en el .gz: los mismos datos dan siempre el mismo archivo.
    ruta.write_bytes(gzip.compress(datos, mtime=0) if ruta.suffix == '.gz' else datos)


def candidatos_openmantra(recalcular: bool = False, avance=print) -> list[dict]:
    """Candidatos de OpenMantra, desde candidatos.csv o medidos de nuevo."""
    filas = [] if recalcular else leer_csv(CANDIDATOS)
    vigente = (
        filas
        and list(filas[0]) == CAMPOS
        and filas[0]['version'] == str(V.VERSION_RASGOS)
    )
    if not vigente:
        avance('Midiendo los candidatos de OpenMantra (unos 15 minutos)…')
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
    """Random forest de OpenCV con los parámetros del detector (siempre el mismo para
    los mismos datos)."""
    cv2.setRNGSeed(SEMILLA)
    modelo = cv2.ml.RTrees_create()
    modelo.setMaxDepth(PROFUNDIDAD)
    modelo.setMinSampleCount(MINIMO_HOJA)
    modelo.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, ARBOLES, 0))
    modelo.setCalculateVarImportance(True)
    modelo.train(X, cv2.ml.ROW_SAMPLE, y)
    return modelo


def ejemplos(filas: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(medidas, etiqueta) para entrenar: 1 si el candidato contiene textos enteros y
    0 si no es texto. Los trozos de un texto se omiten: son letras, pero no conviene
    proponerlos en vez del texto completo."""
    utiles = [f for f in filas if f['enteros'] or f['es_texto'] == '0']
    X, _ = matriz(utiles)
    return X, np.array([int(bool(f['enteros'])) for f in utiles], np.int32)


def entrenar_con(filas: list[dict], propias: list[dict]):
    """Entrena con filas de OpenMantra y de páginas propias, repitiendo estas
    últimas hasta que pesen PESO_PROPIAS de las primeras."""
    X, y = ejemplos(filas)
    if propias:
        Xp, yp = ejemplos(propias)
        veces = min(
            REPETIR_MAXIMO, max(1, round(PESO_PROPIAS * len(X) / max(1, len(Xp))))
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


def elegidas(filas: list[dict], p: np.ndarray, tipo: str = 'original') -> list[int]:
    """Índices de las filas que se propondrían, elegidas en cada página como en la
    aplicación (vision.sin_solapes)."""
    por_pagina: dict = {}
    for i, f in enumerate(filas):
        if f['tipo'] == tipo and p[i] >= V.CONFIANZA_PROPONER:
            por_pagina.setdefault((f['grupo'], f['pagina']), []).append(i)
    salida = []
    for ids in por_pagina.values():
        cajas = []
        for i in ids:
            x, y, w, h = (int(filas[i][k]) for k in ('x', 'y', 'w', 'h'))
            cajas.append((x, y, x + w, y + h))
        salida += [ids[k] for k in V.sin_solapes(cajas, p[ids])]
    return salida


def medir_deteccion(
    filas: list[dict],
    p: np.ndarray,
    umbral: float,
    textos: int,
    tipo: str = 'original',
) -> dict:
    """Precisión, recall y F1 aceptando confianza >= umbral, en las páginas del tipo
    dado, y el techo: el recall si se aceptaran todos los candidatos.

    enteros es la parte de los textos que quedan con al menos LETRAS_ENTERAS de sus
    letras dentro de las regiones aceptadas: es más exigente que el recall, que
    cuenta también un trozo. Con px_texto y px_dibujo (solo en las originales), lo
    que borraría el modo automático.
    """
    propias = [f for f in filas if f['tipo'] == tipo]
    aceptados = [filas[i] for i in elegidas(filas, p, tipo) if p[i] >= umbral]
    cubierto: dict = {}
    for f in aceptados:
        for parte in filter(None, f['cubre'].split(';')):
            texto, valor = parte.split(':')
            clave = (f['grupo'], f['pagina'], texto)
            cubierto[clave] = cubierto.get(clave, 0.0) + float(valor)

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
        'techo': hallados(propias) / max(1, textos),
        'enteros': sum(v >= LETRAS_ENTERAS for v in cubierto.values()) / max(1, textos),
        'f1': 2 * precision * recall / max(1e-9, precision + recall),
        'px_texto': sum(int(f['px_texto']) for f in limpias),
        'px_dibujo': sum(int(f['px_dibujo']) for f in limpias),
    }


def _linea(nombre: str, m: dict) -> str:
    """Resumen de una medición en una línea."""
    return (
        f'  {nombre}: {m["textos"]} textos · candidatos {m["techo"]:.0%} · '
        f'P {m["precision"]:.2f} · R {m["recall"]:.2f} · F1 {m["f1"]:.2f} · '
        f'enteros {m["enteros"]:.0%}'
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
            '\n«candidatos» es el techo: el texto que falta ahí nunca se propone (por '
            'ejemplo, está sobre dibujo) y entrenar no lo arregla. «Enteros»: textos '
            f'con al menos el {LETRAS_ENTERAS:.0%} de sus letras dentro de lo propuesto.'
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
