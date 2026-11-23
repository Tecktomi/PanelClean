"""Evaluación de PanelClean: corre las métricas comprometidas en el Hito 1.

Partes:
1. Pares sintéticos: máscara (IoU/Dice), selector, limpieza (MAE/PSNR/SSIM),
   conservación e hipótesis «selección adaptativa vs. método único».
2. OpenMantra: detección de texto (precisión, recall y F1) y selección por zona
   aproximada («Ajustar al texto»).
3. OpenMantra ampliado a 2048 px: tiempo por página.
4. Regiones reales etiquetadas a mano (si existe evaluacion/etiquetas.csv): selector.

Uso, desde la raíz del repositorio:
    python -m evaluacion.evaluar            # evaluación completa (unos minutos)
    python -m evaluacion.evaluar --rapido   # versión corta para probar
Los resultados quedan en evaluacion/resultados/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from panelclean import __version__
from panelclean import vision as V

from . import cajas as K
from . import calibrar_selector
from . import entrenar_detector as D
from . import metricas as M
from . import muestreo
from . import sinteticos as S

CARPETA = Path(__file__).resolve().parent
RESULTADOS = CARPETA / 'resultados'
DATASET = CARPETA.parent / 'datasets' / 'open-mantra'
ETIQUETAS = CARPETA / 'etiquetas.csv'

METODOS_FIJOS = ('Relleno local', 'Inpainting Telea', 'Inpainting NS')
METODOS = METODOS_FIJOS + ('Automático',)
# Obra reservada como conjunto de prueba (nunca se usa para ajustar nada).
LIBRO_PRUEBA = 'tencho_isoro'
LADO_RENDIMIENTO = 2048
META_TIEMPO = 15.0


def clase_simple(nombre: str) -> str:
    """Nombre de clase de la aplicación → clase de la evaluación."""
    return 'Complejo' if nombre.startswith('Complejo') else nombre


def num(valor: float, decimales: int = 2) -> str:
    """Número con coma decimal, para pegarlo directo en el informe."""
    return f'{valor:.{decimales}f}'.replace('.', ',')


def con_ic(valores, decimales: int = 2) -> str:
    """«media [inf; sup]» con intervalo de confianza del 95 % por bootstrap."""
    media, inf, sup = M.intervalo_bootstrap(valores)
    return f'{num(media, decimales)} [{num(inf, decimales)}; {num(sup, decimales)}]'


def tabla(encabezado: list[str], filas: list[list]) -> str:
    """Tabla en Markdown."""
    lineas = ['| ' + ' | '.join(encabezado) + ' |', '|' + '---|' * len(encabezado)]
    lineas += ['| ' + ' | '.join(str(c) for c in fila) + ' |' for fila in filas]
    return '\n'.join(lineas)


def guardar_csv(nombre: str, filas: list[dict]) -> None:
    """Guarda las filas en resultados/<nombre>."""
    if not filas:
        return
    with open(RESULTADOS / nombre, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


# ---------------------------------------------------------------------------
# 1. Pares sintéticos
# ---------------------------------------------------------------------------


def evaluar_sinteticos(pares: list[S.Par]) -> tuple[list[dict], list[dict]]:
    """Mide máscara, selector y limpieza en cada par. Devuelve (filas_mascara, filas_limpieza)."""
    filas_mascara, filas_limpieza = [], []
    for i, par in enumerate(pares):
        region = V.bounded_region(par.con_texto, par.caja)
        V.generate_mask(par.con_texto, region, V.Options())
        ys = slice(region.y, region.y + region.h)
        xs = slice(region.x, region.x + region.w)
        prediccion = region.mask.copy()
        referencia = (par.referencia[ys, xs] > 0).astype(np.uint8) * 255
        regla = V.classify_background(
            region.entropy, region.dark_level, texture=region.texture
        )[0]
        filas_mascara.append(
            {
                'par': i,
                'clase': par.clase,
                'borde_blanco': int(par.borde_blanco),
                'iou': M.iou(prediccion, referencia),
                'dice': M.dice(prediccion, referencia),
                'cobertura': M.cobertura(prediccion, referencia),
                'exceso': M.exceso(prediccion, referencia),
                'selector': clase_simple(regla),
                'selector_app': region.classification,
                'entropia': region.entropy,
                'tono_oscuro': region.dark_level,
                'textura': region.texture,
                'varianza': region.variance,
                'densidad_bordes': region.edge_density,
                'texto': par.texto,
            }
        )

        def fila(
            mascara_tipo: str, metodo: str, usado: str, salida, mascara, predicha: str
        ) -> dict:
            limpio, entrada = par.limpio[ys, xs], par.con_texto[ys, xs]
            resultado = salida[ys, xs]
            completa = np.zeros(par.referencia.shape, np.uint8)
            completa[ys, xs] = mascara
            cambiados = np.any(salida != par.con_texto, axis=2)
            danio = M.gris(salida) - M.gris(par.limpio)
            return {
                'par': i,
                'clase': par.clase,
                'borde_blanco': int(par.borde_blanco),
                'mascara': mascara_tipo,
                'clase_predicha': predicha,
                'metodo': metodo,
                'usado': usado,
                'sin_aplicar': int(not np.any(resultado != entrada)),
                'mae_region': M.mae(resultado, limpio),
                'mae_texto': M.mae(resultado, limpio, referencia),
                'psnr_region': M.psnr(resultado, limpio),
                'ssim_region': M.ssim(resultado, limpio),
                'px_fuera_mascara': int((cambiados & (completa == 0)).sum()),
                'px_fondo_danado': int(
                    (
                        (np.abs(danio) > S.UMBRAL_REFERENCIA) & (par.referencia == 0)
                    ).sum()
                ),
            }

        vacia = np.zeros_like(prediccion)
        filas_limpieza.append(fila('-', 'Sin limpiar', '-', par.con_texto, vacia, '-'))
        for tipo, mascara in (('sistema', prediccion), ('perfecta', referencia)):
            region.mask = mascara.copy()
            V.describe_background(par.con_texto, region)
            predicha = clase_simple(
                V.classify_background(
                    region.entropy, region.dark_level, texture=region.texture
                )[0]
            )
            for metodo in METODOS:
                salida, usado, _ = V.clean_region(
                    par.con_texto, region, V.Options(method=metodo)
                )
                filas_limpieza.append(
                    fila(tipo, metodo, usado, salida, mascara, predicha)
                )
    return filas_mascara, filas_limpieza


def resumen_mascara(filas: list[dict]) -> tuple[str, float]:
    """Sección de la máscara. Devuelve (texto, IoU medio)."""
    partes = ['## 1. Máscara de texto (pares sintéticos)', '']
    partes.append(
        'Referencia: los píxeles que el texto cambió respecto del fondo limpio '
        f'(diferencia mayor que {S.UMBRAL_REFERENCIA} niveles de gris), borde blanco incluido. '
        'La máscara se genera con los ajustes por defecto de la aplicación. '
        '*Cobertura* = parte del texto que queda dentro de la máscara (lo que falta queda como resto); '
        '*exceso* = parte de la máscara que no era texto.'
    )
    partes.append('')
    filas_tabla = []
    grupos = [(c, [f for f in filas if f['clase'] == c]) for c in S.CLASES]
    grupos += [
        ('Con borde blanco', [f for f in filas if f['borde_blanco']]),
        ('Sin borde blanco', [f for f in filas if not f['borde_blanco']]),
        ('**Todos**', filas),
    ]
    for nombre, g in grupos:
        if not g:
            continue
        filas_tabla.append(
            [
                nombre,
                len(g),
                con_ic([f['iou'] for f in g]),
                num(np.mean([f['dice'] for f in g])),
                num(np.mean([f['cobertura'] for f in g])),
                num(np.mean([f['exceso'] for f in g])),
            ]
        )
    partes.append(
        tabla(
            ['Fondo', 'Pares', 'IoU [IC 95 %]', 'Dice', 'Cobertura', 'Exceso'],
            filas_tabla,
        )
    )
    return '\n'.join(partes), float(np.mean([f['iou'] for f in filas]))


def resumen_selector(filas: list[dict]) -> tuple[str, float]:
    """Sección del selector en sintéticos. Devuelve (texto, macro-F1)."""
    reales = [f['clase'] for f in filas]
    predichas = [f['selector'] for f in filas]
    clases = list(S.CLASES)
    m = M.matriz_confusion(reales, predichas, clases)
    f1 = M.f1_por_clase(reales, predichas, clases)
    macro = M.macro_f1(reales, predichas, clases)
    revisar = sum(f['selector_app'] == 'Revisar máscara' for f in filas)
    partes = ['## 2. Selector de estrategia (pares sintéticos)', '']
    partes.append(
        'La clase real es la del fondo generado. Se evalúa la regla de entropía y '
        'tono oscuro (`classify_background`) sin el aviso de «Revisar máscara», '
        'que no es una clase de fondo. Sus umbrales se ajustaron con otros pares '
        '(semilla 7, ver `calibrar_selector.py`), así que estos pares no se usaron '
        'para ajustarla.'
    )
    partes.append('')
    partes.append(
        tabla(
            ['Real \\ Predicha'] + clases + ['F1'],
            [[c] + list(m[k]) + [num(f1[c])] for k, c in enumerate(clases)],
        )
    )
    partes.append('')
    partes.append(
        f'Exactitud: **{num(M.exactitud(reales, predichas))}** · '
        f'Macro-F1: **{num(macro)}** · '
        f'Regiones con aviso «Revisar máscara» en la aplicación: {revisar} de {len(filas)}.'
    )
    return '\n'.join(partes), macro


# Método que aplica el selector según la clase de fondo. El modo automático de la
# aplicación deja «Complejo» para revisión manual; para poder comparar en todos los
# pares, aquí se usa Telea (el inpainting que la aplicación ya usa por defecto).
MAPA_SELECTOR = {
    'Uniforme': 'Relleno local',
    'Estructurado': 'Inpainting Telea',
    'Complejo': 'Inpainting Telea',
}
# (columna, nombre, sentido, decimales). sentido = 1 si menor es mejor, -1 si mayor es mejor.
CRITERIOS = (
    ('mae_region', 'MAE', 1, 2),
    ('psnr_region', 'PSNR', -1, 1),
    ('ssim_region', 'SSIM', -1, 3),
)


def resumen_limpieza(filas: list[dict]) -> tuple[str, dict]:
    """Secciones de limpieza, hipótesis y conservación. Devuelve (texto, datos clave)."""
    datos = {}
    partes = ['## 3. Limpieza e hipótesis central (pares sintéticos)', '']
    partes.append(
        'Se comparan la salida y el fondo limpio en escala de grises, dentro de la '
        'selección. MAE en niveles de gris (0-255, menor es mejor); PSNR en dB y SSIM '
        '(mayor es mejor). Ojo: el MAE favorece por construcción el relleno con la mediana '
        '(la mediana es el valor que minimiza el error absoluto), aunque borre la textura; '
        'el SSIM sí mide si la textura se conserva. Por eso la hipótesis se juzga con las tres.'
    )
    partes.append('')
    partes.append(
        'Estrategias: los tres métodos únicos; el **adaptativo (selector actual)**, que '
        'usa Relleno local si la regla clasifica el fondo como Uniforme y Telea en otro caso; '
        'el **adaptativo (clase real)**, igual pero con la clase verdadera del fondo (lo que '
        'lograría un clasificador perfecto), y el **oráculo por par**, que elige el mejor '
        'método único de cada par según el MAE (techo). Todas se miden en los mismos pares.'
    )
    por_par: dict = {}
    for f in filas:
        por_par.setdefault((f['mascara'], f['par']), {})[f['metodo']] = f
    sin_limpiar = {f['par']: f for f in filas if f['metodo'] == 'Sin limpiar'}

    for tipo, titulo in (
        ('perfecta', 'Con la máscara perfecta (aísla la elección del método)'),
        ('sistema', 'Con la máscara del sistema (resultado que obtiene el usuario)'),
    ):
        pares = {p: v for (t, p), v in por_par.items() if t == tipo}
        ids = sorted(pares)
        if not ids:
            continue

        def selector(p, pares=pares):
            return pares[p][MAPA_SELECTOR[pares[p]['Automático']['clase_predicha']]]

        def clase_real(p, pares=pares):
            return pares[p][MAPA_SELECTOR[pares[p]['Automático']['clase']]]

        def oraculo(p, pares=pares):
            return min(
                (pares[p][m] for m in METODOS_FIJOS), key=lambda v: v['mae_region']
            )

        estrategias = (
            [('Sin limpiar', lambda p: sin_limpiar[p])]
            + [(m, lambda p, m=m, pares=pares: pares[p][m]) for m in METODOS_FIJOS]
            + [
                ('Adaptativo (selector actual)', selector),
                ('Adaptativo (clase real)', clase_real),
                ('Oráculo por par', oraculo),
            ]
        )
        partes += ['', f'### {titulo}', '']
        partes.append(
            tabla(
                ['Estrategia', 'MAE [IC 95 %]', 'PSNR (dB)', 'SSIM'],
                [
                    [
                        nombre,
                        con_ic([elegir(p)['mae_region'] for p in ids]),
                        num(np.mean([elegir(p)['psnr_region'] for p in ids]), 1),
                        num(np.mean([elegir(p)['ssim_region'] for p in ids]), 3),
                    ]
                    for nombre, elegir in estrategias
                ],
            )
        )
        # Por clase de fondo: qué método conviene en cada una.
        partes += ['', 'Por clase de fondo (MAE · SSIM):', '']
        filas_clase = []
        for clase in S.CLASES:
            ps = [p for p in ids if pares[p]['Automático']['clase'] == clase]
            if ps:
                filas_clase.append(
                    [clase]
                    + [
                        f"{num(np.mean([elegir(p)['mae_region'] for p in ps]))} · "
                        f"{num(np.mean([elegir(p)['ssim_region'] for p in ps]), 3)}"
                        for _, elegir in estrategias[:-1]
                    ]
                )
        partes.append(tabla(['Fondo'] + [n for n, _ in estrategias[:-1]], filas_clase))
        # Hipótesis: adaptativo (selector actual) contra el mejor método único, por métrica.
        partes += [
            '',
            '**Hipótesis** (adaptativo con el selector actual vs. mejor método único):',
            '',
        ]
        veredictos = {}
        for campo, nombre, sentido, dec in CRITERIOS:
            medias = {
                m: np.mean([pares[p][m][campo] for p in ids]) for m in METODOS_FIJOS
            }
            mejor = min(medias, key=lambda m: sentido * medias[m])
            dif = [selector(p)[campo] - pares[p][mejor][campo] for p in ids]
            media, inf, sup = M.intervalo_bootstrap(dif)
            # Mejora = diferencia negativa en MAE y positiva en PSNR/SSIM.
            if max(sentido * inf, sentido * sup) < 0:
                veredicto = 'mejor'
            elif min(sentido * inf, sentido * sup) > 0:
                veredicto = 'peor'
            else:
                veredicto = 'sin diferencia significativa'
            veredictos[nombre] = veredicto
            partes.append(
                f'- {nombre}: mejor método único = {mejor}; adaptativo − {mejor} = '
                f'{num(media, dec)} [{num(inf, dec)}; {num(sup, dec)}] → **{veredicto}**.'
            )
        # Lo que hace de verdad el modo automático de la aplicación.
        derivados = {
            c: sum(
                1
                for p in ids
                if pares[p]['Automático']['clase'] == c
                and pares[p]['Automático']['sin_aplicar']
            )
            for c in S.CLASES
        }
        total = {
            c: sum(1 for p in ids if pares[p]['Automático']['clase'] == c)
            for c in S.CLASES
        }
        partes += [
            '',
            'En la aplicación, el modo automático deja para revisión manual (sin limpiar) '
            + ', '.join(
                f'{derivados[c]} de {total[c]} pares {c.lower()}s' for c in S.CLASES
            )
            + '.',
        ]
        datos[tipo] = veredictos

    fuera = max(f['px_fuera_mascara'] for f in filas)
    danio = [
        f['px_fondo_danado']
        for f in filas
        if f['mascara'] == 'sistema' and f['metodo'] in METODOS_FIJOS
    ]
    partes += ['', '## 4. Conservación', '']
    partes.append(
        f'Píxeles modificados fuera de la máscara, en todas las limpiezas: **máximo {fuera}** '
        '(meta: 0). Dentro de la máscara del sistema, los métodos alteraron de forma visible '
        f'(más de {S.UMBRAL_REFERENCIA} niveles) en promedio {num(np.mean(danio), 1)} píxeles de '
        'fondo sano por par: es el costo de que la máscara marque algo más que el texto.'
    )
    datos['fuera'] = fuera
    return '\n'.join(partes), datos


# ---------------------------------------------------------------------------
# 2. Detección en OpenMantra
# ---------------------------------------------------------------------------


def cargar_openmantra() -> list[dict]:
    """Anotaciones de OpenMantra."""
    return json.loads((DATASET / 'annotation.json').read_text(encoding='utf-8'))


def dibujo_en_modo_automatico(
    imagen: np.ndarray, regiones: list[V.Region], textos: list[tuple]
) -> dict:
    """Qué haría «Limpiar todas» en automático con estas regiones.

    Cuenta las regiones que se limpiarían y las que quedan para revisión, y los
    píxeles de máscara de las que se limpiarían, separando los que caen dentro de
    una caja de texto anotada de los que caen fuera (dibujo que se borraría).
    """
    en_texto = np.zeros(imagen.shape[:2], bool)
    for x1, y1, x2, y2 in textos:
        en_texto[y1:y2, x1:x2] = True
    cuenta = {'auto_limpiadas': 0, 'auto_revision': 0, 'px_texto': 0, 'px_dibujo': 0}
    for r in regiones:
        mascara = V.generate_mask(imagen, r, V.Options()) > 0
        if r.suggested_method == 'Revisión manual':
            cuenta['auto_revision'] += 1
            continue
        cuenta['auto_limpiadas'] += 1
        zona = en_texto[r.y : r.y + r.h, r.x : r.x + r.w]
        cuenta['px_texto'] += int((mascara & zona).sum())
        cuenta['px_dibujo'] += int((mascara & ~zona).sum())
    return cuenta


def evaluar_deteccion(datos: list[dict], por_libro: int | None = None) -> list[dict]:
    """Corre detect_bubbles (el detector anterior) en cada página y cuenta aciertos.

    Se cuentan con el criterio actual (cajas.acierta: el centro cae en el texto y la
    región no mide más de 3 veces su caja) y con el original (cobertura del 80 % de
    la caja). También qué borraría el modo automático si se limpiaran todas.
    """
    filas = []
    for libro in datos:
        for pagina in libro['pages'][:por_libro]:
            imagen = V.bgr_from_file(str(DATASET / pagina['image_paths']['ja']))
            regiones = V.detect_bubbles(imagen)
            detecciones = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in regiones]
            textos = [
                (t['x'], t['y'], t['x'] + t['w'], t['y'] + t['h'])
                for t in pagina['text']
            ]
            acertados, correctas = M.aciertos_por_cobertura(detecciones, textos)
            filas.append(
                {
                    'libro': libro['book_title'],
                    'pagina': pagina['page_index'],
                    'textos': len(textos),
                    'detecciones': len(detecciones),
                    'textos_hallados': sum(
                        any(K.acierta(d, t) for d in detecciones) for t in textos
                    ),
                    'detecciones_correctas': sum(
                        any(K.acierta(d, t) for t in textos) for d in detecciones
                    ),
                    'textos_cobertura': acertados,
                    'correctas_cobertura': correctas,
                    **dibujo_en_modo_automatico(imagen, regiones, textos),
                }
            )
    return filas


def evaluar_detector_texto(
    datos: list[dict], recalcular: bool = True, avance=print
) -> tuple[list[dict], dict]:
    """Mide el detector entrenado en OpenMantra y en las series propias marcadas.

    Cada obra de desarrollo (y cada serie propia) se mide con un modelo entrenado
    sin ella; la obra reservada y las series reservadas, con el modelo final, que no
    las usa. Devuelve (filas por candidato para el CSV, medidas por conjunto).
    """
    filas = D.candidatos_openmantra(recalcular, avance)
    propias = D.medir_propias(avance)
    reservadas = {g for g, serie in K.leer().items() if serie['reservada']}
    p, pp, _ = D.validacion_cruzada(filas, propias, reservadas)
    totales = D.textos_por_grupo(datos)
    grupos = list(dict.fromkeys(f['grupo'] for f in filas))
    conjuntos = {g: {g} for g in grupos}
    conjuntos['desarrollo'] = set(grupos) - {LIBRO_PRUEBA}
    conjuntos['todas'] = set(grupos)
    textos = {
        (libro['book_title'], str(pagina['page_index'])): [
            (t['x'], t['y'], t['x'] + t['w'], t['y'] + t['h']) for t in pagina['text']
        ]
        for libro in datos
        for pagina in libro['pages']
    }
    medidas = {}
    for nombre, incluidos in conjuntos.items():
        sel = [i for i, f in enumerate(filas) if f['grupo'] in incluidos]
        sub, q = [filas[i] for i in sel], p[sel]
        n = sum(totales[g] for g in incluidos)
        medidas[nombre] = {
            'proponer': D.medir_deteccion(sub, q, V.CONFIANZA_PROPONER, n),
            'limpiar': D.medir_deteccion(sub, q, V.CONFIANZA_LIMPIAR, n),
        }
        # Criterio original: cobertura del 80 % de la caja anotada.
        aceptadas: dict = {}
        for f, c in zip(sub, q):
            if f['tipo'] == 'original' and c >= V.CONFIANZA_LIMPIAR:
                x, y, w, h = (int(f[k]) for k in ('x', 'y', 'w', 'h'))
                aceptadas.setdefault((f['grupo'], f['pagina']), []).append(
                    (x, y, x + w, y + h)
                )
        hallados = correctas = 0
        for clave, cajas in textos.items():
            if clave[0] in incluidos:
                a, b = M.aciertos_por_cobertura(aceptadas.get(clave, []), cajas)
                hallados, correctas = hallados + a, correctas + b
        medidas[nombre]['cobertura'] = M.precision_recall_f1(
            correctas, medidas[nombre]['limpiar']['detecciones'], hallados, n
        )
    dev = [i for i, f in enumerate(filas) if f['grupo'] != LIBRO_PRUEBA]
    medidas['horizontal'] = D.medir_deteccion(
        [filas[i] for i in dev],
        p[dev],
        V.CONFIANZA_LIMPIAR,
        sum(totales[g] for g in conjuntos['desarrollo']),
        'horizontal',
    )
    propios = D.textos_propios()
    medidas['propias'] = {}
    for g in dict.fromkeys(f['grupo'] for f in propias):
        sel = [i for i, f in enumerate(propias) if f['grupo'] == g]
        m = D.medir_deteccion(
            [propias[i] for i in sel],
            pp[sel],
            V.CONFIANZA_LIMPIAR,
            propios[g],
            'propia',
        )
        medidas['propias'][g] = [('reservada' if g in reservadas else 'sin verla', m)]
        q = None
        if g not in reservadas:
            q = D.validacion_por_paginas(filas, propias, g)
        if q is not None:
            m = D.medir_deteccion(
                [propias[i] for i in sel], q, V.CONFIANZA_LIMPIAR, propios[g], 'propia'
            )
            medidas['propias'][g].append(('viendo otras de sus páginas', m))
    salida = [
        {
            k: f[k]
            for k in ('grupo', 'pagina', 'x', 'y', 'w', 'h', 'es_texto', 'px_dibujo')
        }
        | {'confianza': round(float(c), 3)}
        for f, c in zip(filas, p)
        if f['tipo'] == 'original'
    ]
    return salida, medidas


GIGANTE = 0.15  # fracción de la hoja desde la que una región es «gigante»


def _aclarado(factor, suma=0.0):
    """Variante de escaneo: multiplica (por canal o en total) y suma un tono."""
    return lambda im: np.clip(
        im.astype(np.float32) * np.array(factor) + suma, 0, 255
    ).astype(np.uint8)


def _sombra(imagen: np.ndarray) -> np.ndarray:
    """Variante de escaneo: la página se oscurece hacia la derecha (hasta un 28 %)."""
    peso = np.linspace(1.0, 0.72, imagen.shape[1])[None, :, None]
    return (imagen.astype(np.float32) * peso).astype(np.uint8)


def _a_tamano(k: float):
    """Variante de escaneo: la página con k veces la resolución."""
    modo = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    return lambda im: cv2.resize(im, None, fx=k, fy=k, interpolation=modo)


VARIANTES = {  # nombre: (cómo se altera la página, escala)
    'Original': (lambda im: im, 1.0),
    'Papel 18 % más oscuro': (_aclarado(0.82), 1.0),
    'Papel amarillento': (_aclarado([0.72, 0.86, 0.92]), 1.0),
    'Tinta lavada': (_aclarado(0.62, 80), 1.0),
    'Sombra hacia un lado': (_sombra, 1.0),
    'Resolución × 2': (_a_tamano(2.0), 2.0),
    'Resolución × 0,6': (_a_tamano(0.6), 0.6),
}


def evaluar_escaneos(datos: list[dict], cada: int = 6) -> list[dict]:
    """Candidatos y clics en páginas de desarrollo alteradas como escaneos distintos.

    No depende del modelo: mide qué textos aparecen entre los candidatos y qué pasa
    al hacer clic en el centro de cada texto (si queda una región correcta y si se
    confirma alguna gigante).
    """
    paginas = [
        pagina
        for libro in datos
        if libro['book_title'] != LIBRO_PRUEBA
        for pagina in libro['pages'][::cada]
    ]
    filas = []
    for nombre, (alterar, k) in VARIANTES.items():
        cuenta = {'textos': 0, 'candidatos': 0, 'clic': 0, 'clic_gigante': 0}
        for pagina in paginas:
            base = V.bgr_from_file(str(DATASET / pagina['image_paths']['ja']))
            imagen = alterar(base)
            preparada = V.preparar_pagina(imagen)
            papel = V.papel_de_pagina(imagen, preparada)
            candidatos, _ = V.candidatos_de_texto(imagen, preparada)
            for r in candidatos:
                r.papel = V._papel_alrededor(papel, (r.x, r.y, r.x + r.w, r.y + r.h))
            limite = GIGANTE * imagen.shape[0] * imagen.shape[1]
            for t in pagina['text']:
                caja = tuple(
                    k * v for v in (t['x'], t['y'], t['x'] + t['w'], t['y'] + t['h'])
                )

                def correcta(r):
                    """La región corresponde a este texto."""
                    return K.acierta((r.x, r.y, r.x + r.w, r.y + r.h), caja)

                punto = (int((caja[0] + caja[2]) / 2), int((caja[1] + caja[3]) / 2))
                confirmadas = V.regiones_del_punto(candidatos, punto, papel)
                if not confirmadas:
                    confirmadas = V.buscar_en_punto(imagen, punto, preparada)
                cuenta['textos'] += 1
                cuenta['candidatos'] += any(map(correcta, candidatos))
                cuenta['clic'] += any(map(correcta, confirmadas))
                cuenta['clic_gigante'] += any(r.w * r.h > limite for r in confirmadas)
        filas.append({'variante': nombre, 'paginas': len(paginas), **cuenta})
    return filas


def miles(n: int) -> str:
    """Entero con punto de miles, como se escribe en español."""
    return f'{n:,}'.replace(',', '.')


def resumen_deteccion(
    filas: list[dict], medidas: dict, escaneos: list[dict]
) -> tuple[str, dict]:
    """Sección de detección. Devuelve (texto, valores para la tabla de metas)."""

    def anterior(g):
        """Precisión, recall y F1 de detect_bubbles con el criterio actual."""
        return M.precision_recall_f1(
            sum(f['detecciones_correctas'] for f in g),
            sum(f['detecciones'] for f in g),
            sum(f['textos_hallados'] for f in g),
            sum(f['textos'] for f in g),
        )

    def cobertura_anterior(g):
        """F1 de detect_bubbles con el criterio original."""
        return M.precision_recall_f1(
            sum(f['correctas_cobertura'] for f in g),
            sum(f['detecciones'] for f in g),
            sum(f['textos_cobertura'] for f in g),
            sum(f['textos'] for f in g),
        )[2]

    partes = ['## 5. Detección de texto (OpenMantra)', '']
    partes.append(
        'Detector de «Detectar texto» y de «Procesar todas»: bloques de tinta rodeados de '
        'papel en la página entera, filtrados por un random forest de OpenCV que mira '
        f'{len(V.RASGOS_TEXTO)} medidas de cada bloque (`entrenar_detector.py`). Se '
        f'informan las regiones con confianza ≥ {num(V.CONFIANZA_LIMPIAR, 1)}, las que el '
        'modo automático limpia. Cada obra de desarrollo se mide con un modelo entrenado '
        'sin ella. **Criterio**: una región acierta un texto si su centro cae en la caja '
        'anotada y no mide más de 3 veces esa caja.'
    )
    partes.append('')
    partes.append(
        f'**Qué no dice esta tabla.** `{LIBRO_PRUEBA}` nunca se usó para entrenar, pero se '
        'consultó varias veces durante el desarrollo (para mirar errores y comparar '
        'alternativas), así que su resultado no es una prueba a ciegas. Y las cinco obras '
        'son del mismo conjunto, todas en japonés: no dice cómo le va en otros idiomas o '
        'estilos. Para eso están las series propias que se marcan con `marcar.py`.'
    )
    partes.append('')
    filas_tabla = []
    conjuntos = [
        (libro, [f for f in filas if f['libro'] == libro])
        for libro in dict.fromkeys(f['libro'] for f in filas)
    ]
    conjuntos.append(('desarrollo', [f for f in filas if f['libro'] != LIBRO_PRUEBA]))
    conjuntos.append(('todas', filas))
    nombres = {
        LIBRO_PRUEBA: f'{LIBRO_PRUEBA} (reservada)',
        'desarrollo': '**Desarrollo**',
        'todas': '**Todas**',
    }
    for clave, g in conjuntos:
        m = medidas[clave]['limpiar']
        filas_tabla.append(
            [
                nombres.get(clave, clave),
                sum(f['textos'] for f in g),
                ' · '.join(num(v) for v in anterior(g)),
                num(m['precision']),
                num(m['recall']),
                f'**{num(m["f1"])}**',
                f'{num(cobertura_anterior(g))} → {num(medidas[clave]["cobertura"][2])}',
            ]
        )
    partes.append(
        tabla(
            [
                'Obra',
                'Textos',
                'Anterior (`detect_bubbles`): P · R · F1',
                'Precisión',
                'Recall',
                'F1',
                'F1 con el criterio original (anterior → actual)',
            ],
            filas_tabla,
        )
    )
    partes.append('')
    partes.append(
        'El criterio original, usado hasta el 23/11, pedía cubrir el 80 % de la caja '
        'anotada. Las cajas de OpenMantra son más holgadas que las letras y las regiones '
        'actuales se ajustan a ellas, así que casi nunca lo cumplen aunque contengan todo '
        'el texto. Se informan los dos: el cambio de criterio favorece al detector actual.'
    )
    partes.append('')
    prop, limp = medidas['todas']['proponer'], medidas['todas']['limpiar']
    limpiados = limp['px_dibujo'] + limp['px_texto']
    dibujo_ant = sum(f['px_dibujo'] for f in filas)
    total_ant = dibujo_ant + sum(f['px_texto'] for f in filas)
    partes.append(
        f'Contando también las dudosas (confianza ≥ {num(V.CONFIANZA_PROPONER, 1)}, en '
        f'naranja, que no se limpian solas), el recall es {num(prop["recall"])} y la '
        f'precisión {num(prop["precision"])}. El techo, si se aceptaran todos los '
        f'candidatos, es {num(limp["techo"])}.'
    )
    partes.append('')
    partes.append(
        '**Dibujo que borraría el modo automático.** De los píxeles que se limpian en '
        f'todas las páginas, {num(100 * limp["px_dibujo"] / max(1, limpiados), 1)} % están '
        f'fuera de las cajas de texto anotadas: {miles(limp["px_dibujo"])} de '
        f'{miles(limpiados)}. Con `detect_bubbles` eran '
        f'{num(100 * dibujo_ant / max(1, total_ant), 0)} % ({miles(dibujo_ant)} de '
        f'{miles(total_ant)}).'
    )
    partes.append('')
    h = medidas['horizontal']
    partes.append(
        '**Texto horizontal simulado.** En las páginas de desarrollo con los globos '
        'traducidos al inglés (`horizontales.py`), cada obra sin verla: precisión '
        f'{num(h["precision"])}, recall {num(h["recall"])} y techo {num(h["techo"])}. Es una '
        'aproximación: las fuentes son las mismas del entrenamiento y no hay páginas reales.'
    )
    partes.append('')
    partes.append(
        f'**Escaneos distintos y clics** ({escaneos[0]["paginas"]} páginas de desarrollo, '
        'sin usar el modelo). «Candidatos»: textos que el detector llega a proponer. '
        '«Clic»: textos que quedan con una región correcta al hacer clic en su centro. '
        f'«Gigantes»: clics que confirman una región de más del {num(100 * GIGANTE, 0)} % '
        'de la hoja.'
    )
    partes.append('')
    partes.append(
        tabla(
            ['Variante', 'Textos', 'Candidatos', 'Clic', 'Gigantes'],
            [
                [
                    e['variante'],
                    e['textos'],
                    f'{num(100 * e["candidatos"] / e["textos"], 0)} %',
                    f'{num(100 * e["clic"] / e["textos"], 0)} %',
                    e['clic_gigante'],
                ]
                for e in escaneos
            ],
        )
    )
    partes.append('')
    if medidas['propias']:
        partes.append(
            '**Series propias** (marcadas con `marcar.py`). «Sin verla»: el modelo no se '
            'entrenó con esa serie. «Viendo otras de sus páginas»: se entrenó con parte de '
            'ellas y se midió en las demás. Si la serie no es reservada, sus cajas partieron '
            'de las propuestas del detector, y eso lo favorece algo.'
        )
        partes.append('')
        partes.append(
            tabla(
                ['Serie', 'Textos', 'Candidatos (techo)', 'Precisión', 'Recall', 'F1'],
                [
                    [
                        f'Serie {k} ({como})',
                        m['textos'],
                        num(m['techo']),
                        num(m['precision']),
                        num(m['recall']),
                        f'**{num(m["f1"])}**',
                    ]
                    for k, casos in enumerate(medidas['propias'].values(), 1)
                    for como, m in casos
                ],
            )
        )
    else:
        partes.append(
            '**Series propias:** todavía no hay páginas marcadas, así que no hay una '
            'medida en otros idiomas.'
        )
    return '\n'.join(partes), {
        'f1': limp['f1'],
        'f1_reservada': medidas[LIBRO_PRUEBA]['limpiar']['f1'],
        'f1_original': medidas['todas']['cobertura'][2],
    }


# ---------------------------------------------------------------------------
# 2b. Selección por zona aproximada («Ajustar al texto»)
# ---------------------------------------------------------------------------

ZONAS_POR_LIBRO = 120
DIBUJO_VISIBLE = 50  # px de dibujo borrado desde los que una zona cuenta como dañina


def _mascara_de(imagen: np.ndarray, cajas: list[tuple]) -> np.ndarray:
    """Unión, del tamaño de la página, de las máscaras de cada caja (x1, y1, x2, y2)."""
    total = np.zeros(imagen.shape[:2], bool)
    for caja in cajas:
        try:
            r = V.bounded_region(imagen, caja)
        except ValueError:
            continue
        total[r.y : r.y + r.h, r.x : r.x + r.w] |= (
            V.generate_mask(imagen, r, V.Options()) > 0
        )
    return total


def evaluar_zonas(
    datos: list[dict], por_libro: int | None = ZONAS_POR_LIBRO
) -> list[dict]:
    """Simula a una persona que encierra cada texto a grandes rasgos y compara dos
    usos de ese rectángulo: como región tal cual, o como zona donde buscar el texto
    (texto_en_zona, la casilla «Ajustar al texto»).

    La zona es la caja anotada más un margen al azar de 10–70 % de su lado corto
    (más 6 px) en cada lado. Las letras de referencia son la máscara que da la caja
    anotada (las cajas de OpenMantra son holgadas y en globos blancos esa máscara
    toma bien las letras). Se mide qué parte de esas letras cubre la máscara de
    cada opción y cuánto borra fuera de todas las cajas anotadas (dibujo).
    """
    filas = []
    for libro in datos:
        nombre = libro['book_title']
        rng = np.random.default_rng([S.SEMILLA, sum(map(ord, nombre))])
        cuenta = 0
        for pagina in libro['pages']:
            if por_libro is not None and cuenta >= por_libro:
                break
            imagen = V.bgr_from_file(str(DATASET / pagina['image_paths']['ja']))
            alto, ancho = imagen.shape[:2]
            textos = [
                (t['x'], t['y'], t['x'] + t['w'], t['y'] + t['h'])
                for t in pagina['text']
            ]
            en_cajas = np.zeros((alto, ancho), bool)
            for x1, y1, x2, y2 in textos:
                en_cajas[y1:y2, x1:x2] = True
            referencia = {t: _mascara_de(imagen, [t]) for t in textos}
            for t in textos:
                if por_libro is not None and cuenta >= por_libro:
                    break
                cuenta += 1
                x1, y1, x2, y2 = t
                m = rng.uniform(0.1, 0.7, 4) * min(x2 - x1, y2 - y1) + 6
                zona = (
                    max(0, int(x1 - m[0])),
                    max(0, int(y1 - m[1])),
                    min(ancho, int(x2 + m[2])),
                    min(alto, int(y2 + m[3])),
                )
                objetivos = [
                    u for u in textos if M._interseccion(u, zona) >= 0.8 * M._area(u)
                ]
                letras = np.zeros((alto, ancho), bool)
                for u in objetivos:
                    letras |= referencia[u]
                if not letras.any():
                    continue
                bloques = [
                    (r.x, r.y, r.x + r.w, r.y + r.h)
                    for r in V.texto_en_zona(imagen, zona)
                ]
                fuera = sum(
                    max(M._interseccion(b, u) for u in textos) < 0.5 * M._area(b)
                    for b in bloques
                )
                fila = {
                    'libro': nombre,
                    'pagina': pagina['page_index'],
                    'zona': zona,
                    'textos': len(objetivos),
                    'bloques': len(bloques),
                    'bloques_fuera': fuera,
                }
                # Sin bloques, la aplicación usa el rectángulo tal cual.
                for modo, cajas in (('zona', [zona]), ('ajuste', bloques or [zona])):
                    mascara = _mascara_de(imagen, cajas)
                    fila[f'cubierto_{modo}'] = float(
                        (mascara & letras).sum() / letras.sum()
                    )
                    fila[f'dibujo_{modo}'] = int((mascara & ~en_cajas).sum())
                filas.append(fila)
    return filas


def resumen_zonas(filas: list[dict]) -> tuple[str, dict]:
    """Sección de «Ajustar al texto». Devuelve (texto, valores de todas las obras)."""

    def medir(g):
        valores = {}
        for modo in ('zona', 'ajuste'):
            valores[modo] = (
                np.mean([f[f'cubierto_{modo}'] for f in g]),
                np.mean([f[f'cubierto_{modo}'] >= 0.95 for f in g]),
                np.mean([f[f'dibujo_{modo}'] for f in g]),
                np.mean([f[f'dibujo_{modo}'] > DIBUJO_VISIBLE for f in g]),
            )
        bloques = sum(f['bloques'] for f in g)
        extra = (
            bloques / sum(f['textos'] for f in g),
            sum(f['bloques_fuera'] for f in g) / max(1, bloques),
            np.mean([f['bloques'] == 0 for f in g]),
        )
        return valores, extra

    def par(a, b, decimales=2, pct=False):
        if pct:
            return f'{num(100 * a, 0)} % → **{num(100 * b, 0)} %**'
        return f'{num(a, decimales)} → **{num(b, decimales)}**'

    filas_tabla = []
    grupos = [
        (
            f'{l} (reservada)' if l == LIBRO_PRUEBA else l,
            [f for f in filas if f['libro'] == l],
        )
        for l in dict.fromkeys(f['libro'] for f in filas)
    ]
    grupos.append(
        (
            '**Desarrollo (sin la reservada)**',
            [f for f in filas if f['libro'] != LIBRO_PRUEBA],
        )
    )
    grupos.append(('**Todas**', filas))
    for nombre, g in grupos:
        v, extra = medir(g)
        z, a = v['zona'], v['ajuste']
        filas_tabla.append(
            [
                nombre,
                len(g),
                par(z[0], a[0]),
                par(z[1], a[1], pct=True),
                par(z[2], a[2], 0),
                par(z[3], a[3], pct=True),
                num(extra[0]),
                f'{num(100 * extra[1], 0)} %',
            ]
        )
    total, extra_total = medir(filas)
    partes = [
        '## 8. Selección por zona aproximada («Ajustar al texto», OpenMantra)',
        '',
        'Simula a una persona que encierra cada texto a grandes rasgos: la caja anotada '
        'más un margen al azar de 10–70 % de su lado corto en cada lado (más 6 px). Se '
        'compara usar ese rectángulo tal cual como región con buscar el texto dentro de '
        'él (`texto_en_zona`); cada celda dice «tal cual → **ajustado**». Las letras de referencia son la '
        'máscara que da la caja anotada; el dibujo borrado son los píxeles de máscara '
        'fuera de todas las cajas anotadas. Los parámetros se ajustaron con las '
        'cuatro obras de desarrollo; la reservada se consultó al desarrollarlo (para ver '
        'sus errores y probar una regla). Si no encuentra texto, la aplicación usa el '
        'rectángulo tal cual, y aquí también.',
        '',
        tabla(
            [
                'Obra',
                'Zonas',
                'Letras cubiertas (media)',
                'Zonas con ≥ 95 % cubierto',
                'Dibujo borrado (px por zona)',
                f'Zonas que borran > {DIBUJO_VISIBLE} px de dibujo',
                'Bloques por texto',
                'Bloques fuera de las cajas',
            ],
            filas_tabla,
        ),
        '',
        f'En {num(100 * extra_total[2], 0)} % de las zonas no encontró texto en globos '
        '(se usa el rectángulo). Un bloque «fuera de las cajas» tiene menos de la mitad '
        'de su área dentro de una caja anotada: casi siempre son trozos de dibujo '
        'encerrados por papel blanco (trazos gruesos, líneas punteadas que decoran el '
        'borde del globo), y a veces texto que OpenMantra no anotó.',
    ]
    return '\n'.join(partes), {
        'dibujo_zona': total['zona'][2],
        'dibujo_ajuste': total['ajuste'][2],
        'cubierto_zona': total['zona'][1],
        'cubierto_ajuste': total['ajuste'][1],
    }


# ---------------------------------------------------------------------------
# 3. Rendimiento
# ---------------------------------------------------------------------------


def nombre_cpu() -> str:
    """Nombre del procesador (en Windows lo lee del registro)."""
    try:
        import winreg

        clave = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'
        )
        return str(winreg.QueryValueEx(clave, 'ProcessorNameString')[0]).strip()
    except Exception:
        return platform.processor() or platform.machine()


def evaluar_rendimiento(datos: list[dict], n_paginas: int) -> list[dict]:
    """Tiempo de detectar el texto, crear las máscaras y limpiar en automático, en
    páginas de OpenMantra ampliadas a 2048 px."""
    paginas = [(libro['book_title'], p) for libro in datos for p in libro['pages']]
    elegidas = np.linspace(0, len(paginas) - 1, min(n_paginas, len(paginas))).astype(
        int
    )
    opciones = V.Options()
    filas = []
    for k in elegidas:
        libro, pagina = paginas[k]
        imagen = V.bgr_from_file(str(DATASET / pagina['image_paths']['ja']))
        escala = LADO_RENDIMIENTO / max(imagen.shape[:2])
        imagen = cv2.resize(
            imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC
        )
        inicio = time.perf_counter()
        regiones = V.detectar_texto(imagen)  # el detector de «Detectar texto»
        trabajo = imagen
        for region in regiones:
            V.generate_mask(imagen, region, opciones)
            trabajo, _, _ = V.clean_region(trabajo, region, opciones)
        segundos = time.perf_counter() - inicio
        filas.append(
            {
                'libro': libro,
                'pagina': pagina['page_index'],
                'ancho': imagen.shape[1],
                'alto': imagen.shape[0],
                'regiones': len(regiones),
                'segundos': round(segundos, 3),
            }
        )
    return filas


def resumen_rendimiento(filas: list[dict]) -> tuple[str, float]:
    """Sección de rendimiento. Devuelve (texto, tiempo máximo)."""
    tiempos = [f['segundos'] for f in filas]
    partes = ['## 6. Rendimiento', '']
    partes.append(
        f'{len(filas)} páginas de OpenMantra repartidas entre las cinco obras, ampliadas a '
        f'{LADO_RENDIMIENTO} px de lado mayor. Se mide detectar regiones, crear sus máscaras y '
        'limpiarlas en modo automático (sin contar abrir el archivo).'
    )
    partes.append('')
    partes.append(
        tabla(
            ['Mediana (s)', 'Máximo (s)', 'Regiones por página (media)', 'Meta'],
            [
                [
                    num(float(np.median(tiempos))),
                    num(max(tiempos)),
                    num(np.mean([f['regiones'] for f in filas]), 1),
                    f'≤ {num(META_TIEMPO, 0)} s',
                ]
            ],
        )
    )
    partes.append('')
    partes.append(
        f'Equipo: {nombre_cpu()}, {os.cpu_count()} hilos, {platform.system()} {platform.release()}.'
    )
    return '\n'.join(partes), max(tiempos)


# ---------------------------------------------------------------------------
# 4. Selector con regiones reales etiquetadas
# ---------------------------------------------------------------------------


def leer_etiquetas() -> tuple[list[dict], dict]:
    """Etiquetas de consenso de etiquetas.csv y datos de acuerdo entre personas.

    Si varias personas etiquetaron la misma región y no coinciden, la región se
    deja fuera hasta que resuelvan el desacuerdo. Las descartadas no se usan.
    """
    if not ETIQUETAS.exists():
        return [], {'dobles': 0, 'acuerdos': 0, 'personas': 0}
    grupos: dict = {}
    personas = set()
    with open(ETIQUETAS, encoding='utf-8') as f:
        for e in csv.DictReader(f):
            personas.add(e['etiquetador'])
            k = (e['libro'], int(e['pagina']), int(e['texto']))
            grupos.setdefault(k, {})[e['etiquetador']] = e['etiqueta']
    consenso, dobles, acuerdos = [], 0, 0
    for (libro, pagina, texto), por_persona in grupos.items():
        valores = set(por_persona.values())
        if len(por_persona) > 1:
            dobles += 1
            acuerdos += len(valores) == 1
        if len(valores) == 1 and (etiqueta := valores.pop()) in S.CLASES:
            consenso.append(
                {'libro': libro, 'pagina': pagina, 'texto': texto, 'etiqueta': etiqueta}
            )
    return consenso, {'dobles': dobles, 'acuerdos': acuerdos, 'personas': len(personas)}


def evaluar_selector_real(datos: list[dict]) -> tuple[list[dict], dict]:
    """Aplica la regla del selector a las regiones etiquetadas (con consenso).

    Cada fila lleva su estrato y su peso N_h / n_h (ver muestreo.py), que corrige
    que las regiones con fondo cargado se mostraran más seguido al etiquetar.
    """
    consenso, acuerdo = leer_etiquetas()
    paginas = {(l['book_title'], p['page_index']): p for l in datos for p in l['pages']}
    estratos = muestreo.cargar_estratos()
    pesos, por_estrato = muestreo.pesos(
        [(e['libro'], e['pagina'], e['texto']) for e in consenso]
    )
    acuerdo['por_estrato'] = por_estrato
    filas, cache = [], {}
    for e in consenso:
        k = (e['libro'], e['pagina'], e['texto'])
        pagina = paginas[(e['libro'], e['pagina'])]
        ruta = DATASET / pagina['image_paths']['ja']
        if ruta not in cache:
            cache[ruta] = V.bgr_from_file(str(ruta))
        t = pagina['text'][e['texto']]
        region = V.bounded_region(
            cache[ruta], (t['x'], t['y'], t['x'] + t['w'], t['y'] + t['h'])
        )
        V.generate_mask(cache[ruta], region, V.Options())
        regla = V.classify_background(
            region.entropy, region.dark_level, texture=region.texture
        )[0]
        filas.append(
            {
                **e,
                'selector': clase_simple(regla),
                'entropia': round(region.entropy, 4),
                'tono_oscuro': region.dark_level,
                'textura': region.texture,
                'estrato': estratos[k],
                'peso': round(pesos[k], 4),
            }
        )
    return filas, acuerdo


def resumen_selector_real(
    filas: list[dict], acuerdo: dict, cv: dict | None
) -> tuple[str, float | None]:
    """Sección del selector con datos reales.

    cv es el resultado de calibrar_selector.validacion_cruzada. Devuelve (texto,
    macro-F1 ponderado de la validación cruzada, o None si no hay etiquetas).
    """
    partes = ['## 7. Selector con regiones reales (OpenMantra, etiquetadas a mano)', '']
    if not filas:
        partes.append(
            'Pendiente: todavía no hay etiquetas. Se etiquetan con '
            '`ETIQUETAR_WINDOWS.bat` (o `python -m evaluacion.etiquetar`) y se '
            'vuelve a correr esta evaluación.'
        )
        return '\n'.join(partes), None
    clases = list(S.CLASES)
    reales = [f['etiqueta'] for f in filas]
    predichas = [f['selector'] for f in filas]
    pesos = [f['peso'] for f in filas]
    m = M.matriz_confusion(reales, predichas, clases)
    f1 = M.f1_por_clase(reales, predichas, clases, pesos)
    macro = M.macro_f1(reales, predichas, clases, pesos)
    partes.append(
        'Casi todo el texto de OpenMantra está en globos blancos, así que al etiquetar '
        'se muestran más seguido las regiones con fondo cargado (muestreo por estratos, '
        'ver `muestreo.py`). La tabla muestra los conteos reales de regiones '
        'etiquetadas; el F1, la exactitud y el macro-F1 están **ponderados** para '
        'estimar el resultado en todo OpenMantra.'
    )
    partes.append('')
    partes.append(
        tabla(
            ['Real \\ Predicha'] + clases + ['F1 (ponderado)'],
            [[c] + list(m[k]) + [num(f1[c])] for k, c in enumerate(clases)],
        )
    )
    partes.append('')
    cuenta = {c: reales.count(c) for c in clases}
    faltan = [c for c in clases if cuenta[c] < 15]
    partes.append(
        f'{len(filas)} regiones, etiquetadas por {acuerdo["personas"]} persona(s). '
        f'Exactitud ponderada: **{num(M.exactitud(reales, predichas, pesos))}** · '
        f'Macro-F1 ponderado: **{num(macro)}** '
        f'(sin ponderar: {num(M.macro_f1(reales, predichas, clases))}).'
        + (
            f' Aún hay pocas regiones de: {", ".join(faltan)} (meta: 15 de cada una).'
            if faltan
            else ''
        )
    )
    partes.append('')
    partes.append(
        'Los umbrales del selector se ajustaron con estas mismas regiones (salvo '
        f'las de la obra de prueba {LIBRO_PRUEBA}, que nunca se usan para ajustar) y '
        'con pares sintéticos de desarrollo (ver `calibrar_selector.py`), así que los '
        'valores de arriba son optimistas. La estimación honesta es la **validación '
        'cruzada por obra**: para cada obra se ajustan los umbrales sin ella y se '
        f'mide en ella. Macro-F1 ponderado: **{num(cv["macro_f1"])}** · F1 por clase: '
        + ' · '.join(f'{c} {num(v)}' for c, v in cv['f1'].items())
        + '. Umbrales de cada ajuste (entropía en bits · tono oscuro): '
        + ' · '.join(
            f'sin {libro} {num(e, 1)} · {t:.0f}'
            for libro, (e, t) in cv['umbrales'].items()
        )
        + '.'
    )
    partes.append('')
    partes.append(
        tabla(
            ['Estrato', 'Etiquetadas', 'Regiones en OpenMantra'],
            [[h, n, N] for h, (n, N) in acuerdo['por_estrato'].items()],
        )
    )
    if acuerdo['dobles']:
        partes.append('')
        partes.append(
            f'{acuerdo["dobles"]} regiones tienen etiquetas de más de una persona; '
            f'coinciden en {acuerdo["acuerdos"]} '
            f'({num(100 * acuerdo["acuerdos"] / acuerdo["dobles"], 0)} %). '
            'Las que no coinciden quedan fuera hasta resolverlas.'
        )
    return '\n'.join(partes), cv['macro_f1']


# ---------------------------------------------------------------------------
# Ejemplos visuales
# ---------------------------------------------------------------------------


def guardar_ejemplos(pares: list[S.Par], por_clase: int = 2) -> None:
    """Lámina con ejemplos: entrada, referencia, máscara, resultado automático y fondo limpio."""
    titulos = ['Con texto', 'Referencia', 'Máscara', 'Automático', 'Fondo limpio']
    fuente = S.fuente(18)
    filas = []
    for clase in S.CLASES:
        for par in [p for p in pares if p.clase == clase][:por_clase]:
            region = V.bounded_region(par.con_texto, par.caja)
            V.generate_mask(par.con_texto, region, V.Options())
            salida, usado, _ = V.clean_region(par.con_texto, region, V.Options())
            x1, y1, x2, y2 = par.caja
            mascara = np.zeros(par.referencia.shape, np.uint8)
            mascara[y1:y2, x1:x2] = region.mask
            entrada = par.con_texto.copy()
            cv2.rectangle(entrada, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
            paneles = [
                entrada,
                cv2.cvtColor(par.referencia, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR),
                salida,
                par.limpio,
            ]
            fila = np.hstack(
                [
                    np.pad(p, ((4, 4), (4, 4), (0, 0)), constant_values=255)
                    for p in paneles
                ]
            )
            imagen = Image.fromarray(cv2.cvtColor(fila, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(imagen).text(
                (10, 8), f'{clase} · {usado}', fill=(200, 0, 0), font=fuente
            )
            filas.append(np.asarray(imagen))
    cabecera = Image.new('RGB', (filas[0].shape[1], 30), 'white')
    dibujo = ImageDraw.Draw(cabecera)
    ancho_panel = filas[0].shape[1] // len(titulos)
    for k, titulo in enumerate(titulos):
        dibujo.text((k * ancho_panel + 10, 6), titulo, fill='black', font=fuente)
    lamina = np.vstack([np.asarray(cabecera)] + filas)
    Image.fromarray(lamina).save(RESULTADOS / 'ejemplos.png')


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Corre todas las partes y escribe resultados/resumen.md."""
    parser = argparse.ArgumentParser(description='Evaluación de PanelClean (Hito 1).')
    parser.add_argument(
        '--rapido', action='store_true', help='versión corta para probar'
    )
    parser.add_argument(
        '--por-clase', type=int, default=50, help='pares sintéticos por clase'
    )
    parser.add_argument(
        '--paginas-tiempo', type=int, default=20, help='páginas para medir tiempo'
    )
    args = parser.parse_args()
    por_clase = 6 if args.rapido else args.por_clase
    por_libro = 3 if args.rapido else None
    n_tiempo = 3 if args.rapido else args.paginas_tiempo
    RESULTADOS.mkdir(exist_ok=True)

    def avance(texto: str) -> None:
        print(f'[{time.strftime("%H:%M:%S")}] {texto}', flush=True)

    avance(f'Generando {por_clase * len(S.CLASES)} pares sintéticos…')
    pares = S.generar_conjunto(por_clase)
    avance('Midiendo máscara, selector y limpieza…')
    filas_mascara, filas_limpieza = evaluar_sinteticos(pares)
    guardar_csv('sinteticos_mascara.csv', filas_mascara)
    guardar_csv('sinteticos_limpieza.csv', filas_limpieza)
    guardar_ejemplos(pares)

    datos = cargar_openmantra()
    avance('Midiendo la detección en OpenMantra…')
    filas_det = evaluar_deteccion(datos, por_libro)
    guardar_csv('deteccion.csv', filas_det)
    avance('Midiendo el detector entrenado…')
    filas_detector, medidas_detector = evaluar_detector_texto(
        datos, recalcular=not args.rapido, avance=avance
    )
    guardar_csv('detector.csv', filas_detector)
    avance('Midiendo escaneos distintos y clics…')
    escaneos = evaluar_escaneos(datos, 24 if args.rapido else 6)
    guardar_csv('escaneos.csv', escaneos)
    avance('Midiendo «Ajustar al texto» con zonas aproximadas…')
    filas_zonas = evaluar_zonas(datos, 10 if args.rapido else ZONAS_POR_LIBRO)
    guardar_csv('zonas.csv', filas_zonas)
    avance(f'Midiendo el tiempo en {n_tiempo} páginas de {LADO_RENDIMIENTO} px…')
    filas_tiempo = evaluar_rendimiento(datos, n_tiempo)
    guardar_csv('rendimiento.csv', filas_tiempo)
    avance('Selector con regiones reales…')
    filas_real, acuerdo = evaluar_selector_real(datos)
    guardar_csv('selector_real.csv', filas_real)
    cv_real = None
    if filas_real:
        avance('Validación cruzada del selector por obra…')
        cv_real = calibrar_selector.validacion_cruzada(
            calibrar_selector.descriptores_sinteticos(),
            calibrar_selector.descriptores_reales(filas_real),
            LIBRO_PRUEBA,
        )

    texto_mascara, iou_medio = resumen_mascara(filas_mascara)
    texto_selector, macro_sint = resumen_selector(filas_mascara)
    texto_limpieza, clave = resumen_limpieza(filas_limpieza)
    texto_det, det = resumen_deteccion(filas_det, medidas_detector, escaneos)
    texto_zonas, zonas = resumen_zonas(filas_zonas)
    texto_tiempo, t_max = resumen_rendimiento(filas_tiempo)
    texto_real, macro_real = resumen_selector_real(filas_real, acuerdo, cv_real)

    def cumple(ok: bool) -> str:
        return 'Sí' if ok else 'No'

    hip = clave.get('sistema', {})
    filas_meta = [
        [
            'Detección en OpenMantra (F1, criterio actual: centro y tamaño)',
            '≥ 0,85',
            num(det['f1']),
            cumple(det['f1'] >= 0.85),
        ],
        [
            f'Detección (F1, criterio actual, obra reservada {LIBRO_PRUEBA}, '
            'consultada durante el desarrollo)',
            '≥ 0,85',
            num(det['f1_reservada']),
            cumple(det['f1_reservada'] >= 0.85),
        ],
        [
            'Detección en OpenMantra (F1, criterio original: cubrir 80 % de la caja)',
            '≥ 0,85',
            num(det['f1_original']),
            cumple(det['f1_original'] >= 0.85),
        ],
        [
            'Máscara (IoU medio, sintéticos)',
            '≥ 0,80',
            num(iou_medio),
            cumple(iou_medio >= 0.80),
        ],
        [
            'Selector (macro-F1, sintéticos)',
            '≥ 0,80',
            num(macro_sint),
            cumple(macro_sint >= 0.80),
        ],
        [
            'Selector (macro-F1, regiones reales, validación cruzada por obra)',
            '≥ 0,80',
            num(macro_real) if macro_real is not None else 'pendiente',
            cumple(macro_real >= 0.80) if macro_real is not None else '—',
        ],
        [
            '«Ajustar al texto»: zonas con ≥ 95 % de las letras cubiertas '
            '(rectángulo tal cual → ajustado)',
            'sin meta',
            f'{num(100 * zonas["cubierto_zona"], 0)} % → '
            f'{num(100 * zonas["cubierto_ajuste"], 0)} %',
            '—',
        ],
        [
            '«Ajustar al texto»: dibujo borrado por zona (rectángulo tal cual → ajustado)',
            'sin meta',
            f'{num(zonas["dibujo_zona"], 0)} px → {num(zonas["dibujo_ajuste"], 0)} px',
            '—',
        ],
        ['Reconstrucción (MAE/PSNR/SSIM)', 'sin meta', 'ver sección 3', '—'],
        [
            'Conservación (píxeles fuera de la máscara)',
            '0',
            str(clave['fuera']),
            cumple(clave['fuera'] == 0),
        ],
        [
            'Rendimiento (máximo por página de 2048 px)',
            '≤ 15 s',
            f'{num(t_max)} s',
            cumple(t_max <= META_TIEMPO),
        ],
        [
            'Hipótesis: adaptativo vs. mejor método único (máscara perfecta)',
            'adaptativo mejor',
            ' · '.join(f'{k}: {v}' for k, v in clave.get('perfecta', {}).items()),
            '—',
        ],
        [
            'Hipótesis: adaptativo vs. mejor método único (máscara del sistema)',
            'adaptativo mejor',
            ' · '.join(f'{k}: {v}' for k, v in hip.items()),
            '—',
        ],
    ]
    encabezado = [
        '# Resultados de la evaluación de PanelClean',
        '',
        f'Generado con `python -m evaluacion.evaluar{" --rapido" if args.rapido else ""}` · '
        f'PanelClean {__version__} · Python {platform.python_version()} · '
        f'OpenCV {cv2.__version__} · NumPy {np.__version__} · '
        f'{len(pares)} pares sintéticos (semilla {S.SEMILLA}) · {len(filas_det)} páginas de OpenMantra.',
        '',
        'Los intervalos entre corchetes son de confianza del 95 % (bootstrap). '
        'Los datos de cada par y cada página están en los CSV de esta carpeta; '
        '`ejemplos.png` muestra algunos pares.',
        '',
        '## Resumen frente a las metas del Hito 1',
        '',
        tabla(['Métrica', 'Meta', 'Resultado', '¿Cumple?'], filas_meta),
    ]
    resumen = '\n\n'.join(
        [
            '\n'.join(encabezado),
            texto_mascara,
            texto_selector,
            texto_limpieza,
            texto_det,
            texto_tiempo,
            texto_real,
            texto_zonas,
        ]
    )
    (RESULTADOS / 'resumen.md').write_text(resumen + '\n', encoding='utf-8')
    avance(f'Listo: {RESULTADOS / "resumen.md"}')


if __name__ == '__main__':
    sys.exit(main())
