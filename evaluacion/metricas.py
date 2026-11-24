"""Métricas de evaluación de PanelClean (sin dependencias fuera de NumPy y OpenCV).

- Máscaras: IoU, Dice, cobertura y exceso.
- Limpieza: MAE, PSNR y SSIM en escala de grises.
- Detección: precisión, recall y F1 con dos criterios de acierto.
- Selector: exactitud, macro-F1 y matriz de confusión.
- Intervalos de confianza por bootstrap.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

Caja = tuple[float, float, float, float]  # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Máscaras
# ---------------------------------------------------------------------------


def _conteos(prediccion: np.ndarray, referencia: np.ndarray) -> tuple[int, int, int]:
    """Devuelve (aciertos, píxeles predichos, píxeles de referencia)."""
    p = np.asarray(prediccion) > 0
    r = np.asarray(referencia) > 0
    return int((p & r).sum()), int(p.sum()), int(r.sum())


def iou(prediccion: np.ndarray, referencia: np.ndarray) -> float:
    """Intersección sobre unión de dos máscaras. Ambas vacías cuenta como 1."""
    comun, n_p, n_r = _conteos(prediccion, referencia)
    union = n_p + n_r - comun
    return 1.0 if union == 0 else comun / union


def dice(prediccion: np.ndarray, referencia: np.ndarray) -> float:
    """Coeficiente de Dice (2·|P∩R| / (|P|+|R|)). Ambas vacías cuenta como 1."""
    comun, n_p, n_r = _conteos(prediccion, referencia)
    return 1.0 if n_p + n_r == 0 else 2 * comun / (n_p + n_r)


def cobertura(prediccion: np.ndarray, referencia: np.ndarray) -> float:
    """Fracción del texto que quedó dentro de la máscara (lo que no, queda como resto)."""
    comun, _, n_r = _conteos(prediccion, referencia)
    return 1.0 if n_r == 0 else comun / n_r


def exceso(prediccion: np.ndarray, referencia: np.ndarray) -> float:
    """Fracción de la máscara que no era texto (fondo que se va a repintar sin necesidad)."""
    comun, n_p, _ = _conteos(prediccion, referencia)
    return 0.0 if n_p == 0 else (n_p - comun) / n_p


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------


def gris(imagen: np.ndarray) -> np.ndarray:
    """Imagen en escala de grises como float64 (acepta BGR o gris)."""
    if imagen.ndim == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen.astype(np.float64)


def mae(a: np.ndarray, b: np.ndarray, zona: Optional[np.ndarray] = None) -> float:
    """Error absoluto medio en niveles de gris (0 a 255), opcionalmente solo en zona."""
    diferencia = np.abs(gris(a) - gris(b))
    if zona is not None:
        diferencia = diferencia[np.asarray(zona) > 0]
    return float(diferencia.mean()) if diferencia.size else 0.0


def psnr(
    a: np.ndarray, b: np.ndarray, zona: Optional[np.ndarray] = None, tope: float = 60.0
) -> float:
    """PSNR en dB con pico 255. Si las imágenes son iguales se devuelve el tope."""
    diferencia = gris(a) - gris(b)
    if zona is not None:
        diferencia = diferencia[np.asarray(zona) > 0]
    mse = float((diferencia**2).mean()) if diferencia.size else 0.0
    if mse == 0:
        return tope
    return min(tope, 10 * np.log10(255.0**2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM medio entre dos imágenes (Wang et al., 2004).

    Ventana gaussiana de 11 × 11 con sigma 1,5 y constantes K1 = 0,01 y K2 = 0,03,
    los valores del artículo original. Vale 1 si las imágenes son iguales.
    """
    x, y = gris(a), gris(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def suave(m: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(m, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)

    mx, my = suave(x), suave(y)
    vx = suave(x * x) - mx * mx
    vy = suave(y * y) - my * my
    cxy = suave(x * y) - mx * my
    mapa = ((2 * mx * my + c1) * (2 * cxy + c2)) / (
        (mx**2 + my**2 + c1) * (vx + vy + c2)
    )
    return float(mapa.mean())


# ---------------------------------------------------------------------------
# Detección
# ---------------------------------------------------------------------------


def _area(c: Caja) -> float:
    """Área de una caja (x1, y1, x2, y2)."""
    return max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])


def _interseccion(a: Caja, b: Caja) -> float:
    """Área común de dos cajas."""
    ancho = min(a[2], b[2]) - max(a[0], b[0])
    alto = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, ancho) * max(0.0, alto)


def iou_cajas(a: Caja, b: Caja) -> float:
    """IoU de dos cajas (x1, y1, x2, y2)."""
    comun = _interseccion(a, b)
    union = _area(a) + _area(b) - comun
    return comun / union if union > 0 else 0.0


def aciertos_por_cobertura(
    detecciones: Sequence[Caja], referencias: Sequence[Caja], minimo: float = 0.8
) -> tuple[int, int]:
    """Criterio principal de PanelClean: lo que importa es que el texto quede dentro.

    Una caja de texto está acertada si al menos `minimo` de su área queda dentro
    de alguna detección. Una detección es correcta si contiene así al menos una
    caja de texto. Devuelve (cajas de texto acertadas, detecciones correctas).
    """

    def cubre(d: Caja, r: Caja) -> bool:
        return _area(r) > 0 and _interseccion(d, r) / _area(r) >= minimo

    textos = sum(any(cubre(d, r) for d in detecciones) for r in referencias)
    correctas = sum(any(cubre(d, r) for r in referencias) for d in detecciones)
    return int(textos), int(correctas)


def aciertos_por_iou(
    detecciones: Sequence[Caja], referencias: Sequence[Caja], umbral: float = 0.5
) -> int:
    """Criterio estándar: emparejamiento uno a uno con IoU ≥ umbral (de mayor a menor IoU)."""
    pares = sorted(
        (
            (iou_cajas(d, r), i, j)
            for i, d in enumerate(detecciones)
            for j, r in enumerate(referencias)
        ),
        reverse=True,
    )
    usadas_d, usadas_r, aciertos = set(), set(), 0
    for valor, i, j in pares:
        if valor < umbral:
            break
        if i not in usadas_d and j not in usadas_r:
            usadas_d.add(i)
            usadas_r.add(j)
            aciertos += 1
    return aciertos


def precision_recall_f1(
    correctas: int, detecciones: int, acertadas: int, referencias: int
) -> tuple[float, float, float]:
    """Precisión = correctas/detecciones, recall = acertadas/referencias y su F1."""
    p = correctas / detecciones if detecciones else 0.0
    r = acertadas / referencias if referencias else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def matriz_confusion(
    reales: Sequence[str],
    predichas: Sequence[str],
    clases: Sequence[str],
    pesos: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Matriz con filas = clase real y columnas = clase predicha.

    Con pesos, cada región suma su peso en vez de 1 (ver muestreo.py): la matriz
    estima entonces los conteos de toda la población.
    """
    indice = {c: k for k, c in enumerate(clases)}
    matriz = np.zeros((len(clases), len(clases)), float if pesos is not None else int)
    if pesos is None:
        pesos = [1] * len(reales)
    for r, p, w in zip(reales, predichas, pesos):
        matriz[indice[r], indice[p]] += w
    return matriz


def exactitud(
    reales: Sequence[str],
    predichas: Sequence[str],
    pesos: Optional[Sequence[float]] = None,
) -> float:
    """Fracción (ponderada, si hay pesos) de regiones bien clasificadas."""
    if not reales:
        return 0.0
    w = np.ones(len(reales)) if pesos is None else np.asarray(pesos, float)
    aciertos = np.array([r == p for r, p in zip(reales, predichas)], float)
    return float((w * aciertos).sum() / w.sum())


def f1_por_clase(
    reales: Sequence[str],
    predichas: Sequence[str],
    clases: Sequence[str],
    pesos: Optional[Sequence[float]] = None,
) -> dict[str, float]:
    """F1 de cada clase (una contra el resto), ponderado si hay pesos."""
    m = matriz_confusion(reales, predichas, clases, pesos)
    salida = {}
    for k, c in enumerate(clases):
        vp = m[k, k]
        predichos, reales_c = m[:, k].sum(), m[k, :].sum()
        p = vp / predichos if predichos else 0.0
        r = vp / reales_c if reales_c else 0.0
        salida[c] = float(2 * p * r / (p + r)) if p + r else 0.0
    return salida


def macro_f1(
    reales: Sequence[str],
    predichas: Sequence[str],
    clases: Sequence[str],
    pesos: Optional[Sequence[float]] = None,
) -> float:
    """Promedio simple del F1 de cada clase: todas pesan igual aunque haya pocas regiones.

    Una clase que no aparece ni en las reales ni en las predichas no se promedia
    (no hay nada que medir en ella).
    """
    f1 = f1_por_clase(reales, predichas, clases, pesos)
    presentes = [c for c in clases if c in reales or c in predichas]
    return float(np.mean([f1[c] for c in presentes])) if presentes else 0.0


def kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Kappa de Cohen entre dos series de etiquetas: 1 = iguales, 0 = lo esperable
    por azar. A diferencia del % de coincidencia, no sube porque una clase sea muy
    común."""
    if not a:
        return 0.0
    n = len(a)
    observado = sum(x == y for x, y in zip(a, b)) / n
    azar = sum(list(a).count(c) * list(b).count(c) for c in set(a) | set(b)) / n**2
    return 1.0 if azar == 1 else (observado - azar) / (1 - azar)


# ---------------------------------------------------------------------------
# Intervalos de confianza
# ---------------------------------------------------------------------------


def intervalo_bootstrap(
    valores: Sequence[float],
    repeticiones: int = 2000,
    nivel: float = 0.95,
    semilla: int = 0,
) -> tuple[float, float, float]:
    """Media e intervalo de confianza por bootstrap (percentiles). Devuelve (media, inf, sup)."""
    v = np.asarray(valores, float)
    if v.size == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(semilla)
    medias = v[rng.integers(0, v.size, (repeticiones, v.size))].mean(axis=1)
    cola = (1 - nivel) / 2 * 100
    inf, sup = np.percentile(medias, [cola, 100 - cola])
    return float(v.mean()), float(inf), float(sup)
