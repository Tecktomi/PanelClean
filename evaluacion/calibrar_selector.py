"""Calibra los dos umbrales del selector de fondo (ENTROPIA_LISO y TONO_TINTA).

La regla está en panelclean/vision.py (classify_background):
- entropía del fondo < ENTROPIA_LISO (y sin trama repartida) -> Uniforme
- si no, tono oscuro (percentil 10) < TONO_TINTA             -> Complejo
- si no                                                      -> Estructurado
El tercer umbral, TEXTURA_REPARTIDA, no cambia ninguna región de desarrollo y
por eso no se ajusta aquí (ver el comentario en vision.py).

Los umbrales se ajustan con datos de desarrollo: pares sintéticos con otra
semilla (7, no la 2026 de la evaluación) y las regiones reales etiquetadas de
las obras que no son de prueba (tencho_isoro está reservada y nunca se usa
para ajustar). Se prueban todas las combinaciones de una grilla y se elige la
que maximiza el promedio de dos macro-F1: el de los sintéticos y el ponderado
de las reales.

Como las regiones reales sirven a la vez para ajustar y para medir, su macro-F1
con los umbrales elegidos es optimista. La estimación honesta es la validación
cruzada por obra: para cada obra se ajustan los umbrales sin ella (y siempre
sin la de prueba) y se mide en ella. La evaluación (evaluar.py) informa ese valor.

Uso, desde la raíz del repositorio:
    python -m evaluacion.calibrar_selector
Si los umbrales que propone cambian, hay que copiarlos en panelclean/vision.py.
"""

from __future__ import annotations

import numpy as np

from panelclean import vision as V

from . import metricas as M
from . import sinteticos as S

SEMILLA_DESARROLLO = 7
POR_CLASE = 60
GRILLA_ENTROPIA = np.round(np.arange(0.2, 3.01, 0.1), 1)  # bits
GRILLA_TONO = np.arange(10, 151, 5)  # niveles de gris
CLASES = list(S.CLASES)


def descriptores_sinteticos(
    semilla: int = SEMILLA_DESARROLLO, por_clase: int = POR_CLASE
) -> list[dict]:
    """Entropía y tono oscuro de pares sintéticos de desarrollo, con su clase real."""
    filas = []
    for par in S.generar_conjunto(por_clase, semilla=semilla):
        region = V.bounded_region(par.con_texto, par.caja)
        V.generate_mask(par.con_texto, region, V.Options())
        filas.append(
            {
                'libro': 'sintético',
                'clase': par.clase,
                'peso': 1.0,
                'entropia': region.entropy,
                'tono_oscuro': region.dark_level,
                'textura': region.texture,
            }
        )
    return filas


def descriptores_reales(filas_selector: list[dict]) -> list[dict]:
    """Adapta las filas de evaluar.evaluar_selector_real al formato de la calibración."""
    return [
        {
            'libro': f['libro'],
            'clase': f['etiqueta'],
            'peso': f['peso'],
            'entropia': f['entropia'],
            'tono_oscuro': f['tono_oscuro'],
            'textura': f['textura'],
        }
        for f in filas_selector
    ]


def predecir(filas: list[dict], umbrales: tuple[float, float]) -> list[str]:
    """Clase que da la regla de la aplicación con los umbrales indicados."""
    predichas = []
    for f in filas:
        nombre = V.classify_background(
            f['entropia'], f['tono_oscuro'], texture=f['textura'], umbrales=umbrales
        )[0]
        predichas.append('Complejo' if nombre.startswith('Complejo') else nombre)
    return predichas


def macro_f1(filas: list[dict], umbrales: tuple[float, float]) -> float:
    """Macro-F1 ponderado (con el peso de cada fila) de la regla con esos umbrales."""
    return M.macro_f1(
        [f['clase'] for f in filas],
        predecir(filas, umbrales),
        CLASES,
        [f['peso'] for f in filas],
    )


def ajustar(sinteticos: list[dict], reales: list[dict]) -> tuple[float, float]:
    """Umbrales de la grilla que maximizan el promedio de ambos macro-F1.

    Si varias combinaciones empatan, se elige la más cercana al centro de las
    empatadas: así el umbral queda lejos de los casos límite.
    """
    puntajes = np.array(
        [
            [
                (macro_f1(sinteticos, (e, t)) + macro_f1(reales, (e, t))) / 2
                for t in GRILLA_TONO
            ]
            for e in GRILLA_ENTROPIA
        ]
    )
    mejores = np.argwhere(puntajes >= puntajes.max() - 1e-9)
    centro = mejores.mean(axis=0)
    i, j = mejores[np.argmin(((mejores - centro) ** 2).sum(axis=1))]
    return float(GRILLA_ENTROPIA[i]), float(GRILLA_TONO[j])


def validacion_cruzada(
    sinteticos: list[dict], reales: list[dict], reservada: str | None = None
) -> dict:
    """Ajusta sin cada obra y predice esa obra. Devuelve macro-F1, F1 por clase y umbrales.

    La obra reservada (la de prueba) también se predice, pero nunca entra en un
    ajuste: para ella se usan los umbrales finales, ajustados con las demás.
    """
    predichas, probadas, umbrales = [], [], {}
    for libro in sorted({f['libro'] for f in reales}):
        prueba = [f for f in reales if f['libro'] == libro]
        umbrales[libro] = ajustar(
            sinteticos,
            [f for f in reales if f['libro'] not in (libro, reservada)],
        )
        predichas += predecir(prueba, umbrales[libro])
        probadas += prueba
    reales_ = [f['clase'] for f in probadas]
    pesos = [f['peso'] for f in probadas]
    return {
        'macro_f1': M.macro_f1(reales_, predichas, CLASES, pesos),
        'f1': M.f1_por_clase(reales_, predichas, CLASES, pesos),
        'umbrales': umbrales,
    }


def main() -> None:
    """Calcula los descriptores, ajusta los umbrales y los compara con los de vision.py."""
    from .evaluar import LIBRO_PRUEBA, cargar_openmantra, evaluar_selector_real

    print(f'Pares sintéticos de desarrollo (semilla {SEMILLA_DESARROLLO})…', flush=True)
    sinteticos = descriptores_sinteticos()
    print('Regiones reales etiquetadas…', flush=True)
    reales = descriptores_reales(evaluar_selector_real(cargar_openmantra())[0])
    desarrollo = [f for f in reales if f['libro'] != LIBRO_PRUEBA]

    actuales = (V.ENTROPIA_LISO, V.TONO_TINTA)
    propuestos = ajustar(sinteticos, desarrollo)
    for nombre, u in (('vision.py', actuales), ('propuestos', propuestos)):
        print(
            f'{nombre:11s} entropía < {u[0]:.1f} bits → Uniforme; '
            f'tono oscuro < {u[1]:.0f} → Complejo   '
            f'macro-F1 sintéticos {macro_f1(sinteticos, u):.3f} · '
            f'reales sin la obra de prueba {macro_f1(desarrollo, u):.3f} (optimista)'
        )
    cv = validacion_cruzada(sinteticos, reales, LIBRO_PRUEBA)
    print(
        f'Validación cruzada por obra (reales): macro-F1 {cv["macro_f1"]:.3f} · '
        + ' · '.join(f'{c} {v:.2f}' for c, v in cv['f1'].items())
    )
    for libro, (e, t) in cv['umbrales'].items():
        print(f'  sin {libro:16s} entropía {e:.1f} · tono {t:.0f}')
    if propuestos != actuales:
        print(
            'Los umbrales propuestos difieren: cópialos en ENTROPIA_LISO y '
            'TONO_TINTA (panelclean/vision.py).'
        )


if __name__ == '__main__':
    main()
