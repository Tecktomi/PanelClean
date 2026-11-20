# Resultados de la evaluación de PanelClean

Generado con `python -m evaluacion.evaluar` · PanelClean 0.3.1-beta · Python 3.15.2 · OpenCV 4.14.0 · NumPy 2.5.5 · 150 pares sintéticos (semilla 2026) · 214 páginas de OpenMantra.

Los intervalos entre corchetes son de confianza del 95 % (bootstrap). Los datos de cada par y cada página están en los CSV de esta carpeta; `ejemplos.png` muestra algunos pares.

## Resumen frente a las metas del Hito 1

| Métrica | Meta | Resultado | ¿Cumple? |
|---|---|---|---|
| Detección (F1, todas las obras) | ≥ 0,85 | 0,54 | No |
| Detección (F1, obra de prueba tencho_isoro) | ≥ 0,85 | 0,68 | No |
| Máscara (IoU medio, sintéticos) | ≥ 0,80 | 0,63 | No |
| Selector (macro-F1, sintéticos) | ≥ 0,80 | 0,83 | Sí |
| Selector (macro-F1, regiones reales, validación cruzada por obra) | ≥ 0,80 | 0,61 | No |
| Reconstrucción (MAE/PSNR/SSIM) | sin meta | ver sección 3 | — |
| Conservación (píxeles fuera de la máscara) | 0 | 0 | Sí |
| Rendimiento (máximo por página de 2048 px) | ≤ 15 s | 1,20 s | Sí |
| Hipótesis: adaptativo vs. mejor método único (máscara perfecta) | adaptativo mejor | MAE: peor · PSNR: mejor · SSIM: peor | — |
| Hipótesis: adaptativo vs. mejor método único (máscara del sistema) | adaptativo mejor | MAE: sin diferencia significativa · PSNR: sin diferencia significativa · SSIM: sin diferencia significativa | — |

## 1. Máscara de texto (pares sintéticos)

Referencia: los píxeles que el texto cambió respecto del fondo limpio (diferencia mayor que 8 niveles de gris), borde blanco incluido. La máscara se genera con los ajustes por defecto de la aplicación. *Cobertura* = parte del texto que queda dentro de la máscara (lo que falta queda como resto); *exceso* = parte de la máscara que no era texto.

| Fondo | Pares | IoU [IC 95 %] | Dice | Cobertura | Exceso |
|---|---|---|---|---|---|
| Uniforme | 50 | 0,98 [0,98; 0,98] | 0,99 | 0,99 | 0,01 |
| Estructurado | 50 | 0,64 [0,58; 0,70] | 0,75 | 0,83 | 0,22 |
| Complejo | 50 | 0,29 [0,23; 0,34] | 0,40 | 0,40 | 0,43 |
| Con borde blanco | 44 | 0,45 [0,41; 0,49] | 0,61 | 0,62 | 0,33 |
| Sin borde blanco | 106 | 0,71 [0,65; 0,78] | 0,76 | 0,79 | 0,18 |
| **Todos** | 150 | 0,63 [0,58; 0,69] | 0,71 | 0,74 | 0,22 |

## 2. Selector de estrategia (pares sintéticos)

La clase real es la del fondo generado. Se evalúa la regla de entropía y tono oscuro (`classify_background`) sin el aviso de «Revisar máscara», que no es una clase de fondo. Sus umbrales se ajustaron con otros pares (semilla 7, ver `calibrar_selector.py`), así que estos pares no se usaron para ajustarla.

| Real \ Predicha | Uniforme | Estructurado | Complejo | F1 |
|---|---|---|---|---|
| Uniforme | 50 | 0 | 0 | 0,92 |
| Estructurado | 4 | 41 | 5 | 0,80 |
| Complejo | 5 | 11 | 34 | 0,76 |

Exactitud: **0,83** · Macro-F1: **0,83** · Regiones con aviso «Revisar máscara» en la aplicación: 12 de 150.

## 3. Limpieza e hipótesis central (pares sintéticos)

Se comparan la salida y el fondo limpio en escala de grises, dentro de la selección. MAE en niveles de gris (0-255, menor es mejor); PSNR en dB y SSIM (mayor es mejor). Ojo: el MAE favorece por construcción el relleno con la mediana (la mediana es el valor que minimiza el error absoluto), aunque borre la textura; el SSIM sí mide si la textura se conserva. Por eso la hipótesis se juzga con las tres.

Estrategias: los tres métodos únicos; el **adaptativo (selector actual)**, que usa Relleno local si la regla clasifica el fondo como Uniforme y Telea en otro caso; el **adaptativo (clase real)**, igual pero con la clase verdadera del fondo (lo que lograría un clasificador perfecto), y el **oráculo por par**, que elige el mejor método único de cada par según el MAE (techo). Todas se miden en los mismos pares.

### Con la máscara perfecta (aísla la elección del método)

| Estrategia | MAE [IC 95 %] | PSNR (dB) | SSIM |
|---|---|---|---|
| Sin limpiar | 30,71 [29,27; 32,21] | 10,8 | 0,633 |
| Relleno local | 8,58 [7,12; 10,09] | 29,3 | 0,866 |
| Inpainting Telea | 6,53 [5,56; 7,55] | 30,8 | 0,900 |
| Inpainting NS | 6,39 [5,43; 7,38] | 30,6 | 0,902 |
| Adaptativo (selector actual) | 6,51 [5,53; 7,53] | 31,3 | 0,901 |
| Adaptativo (clase real) | 6,51 [5,53; 7,53] | 31,3 | 0,901 |
| Oráculo por par | 5,97 [5,05; 6,93] | 31,2 | 0,899 |

Por clase de fondo (MAE · SSIM):

| Fondo | Sin limpiar | Relleno local | Inpainting Telea | Inpainting NS | Adaptativo (selector actual) | Adaptativo (clase real) |
|---|---|---|---|---|---|---|
| Uniforme | 32,19 · 0,535 | 0,21 · 0,992 | 0,28 · 0,989 | 0,29 · 0,990 | 0,21 · 0,992 | 0,21 · 0,992 |
| Estructurado | 30,23 · 0,691 | 8,88 · 0,813 | 9,83 · 0,840 | 9,68 · 0,843 | 9,82 · 0,840 | 9,83 · 0,840 |
| Complejo | 29,71 · 0,674 | 16,64 · 0,793 | 9,49 · 0,871 | 9,22 · 0,873 | 9,51 · 0,870 | 9,49 · 0,871 |

**Hipótesis** (adaptativo con el selector actual vs. mejor método único):

- MAE: mejor método único = Inpainting NS; adaptativo − Inpainting NS = 0,12 [0,06; 0,18] → **peor**.
- PSNR: mejor método único = Inpainting Telea; adaptativo − Inpainting Telea = 0,6 [0,4; 0,7] → **mejor**.
- SSIM: mejor método único = Inpainting NS; adaptativo − Inpainting NS = -0,001 [-0,003; -0,000] → **peor**.

En la aplicación, el modo automático deja para revisión manual (sin limpiar) 0 de 50 pares uniformes, 5 de 50 pares estructurados, 38 de 50 pares complejos.

### Con la máscara del sistema (resultado que obtiene el usuario)

| Estrategia | MAE [IC 95 %] | PSNR (dB) | SSIM |
|---|---|---|---|
| Sin limpiar | 30,71 [29,27; 32,21] | 10,8 | 0,633 |
| Relleno local | 17,60 [14,41; 20,86] | 25,9 | 0,744 |
| Inpainting Telea | 17,90 [14,79; 21,21] | 25,8 | 0,752 |
| Inpainting NS | 17,85 [14,74; 21,11] | 25,4 | 0,753 |
| Adaptativo (selector actual) | 17,81 [14,68; 21,12] | 26,1 | 0,752 |
| Adaptativo (clase real) | 17,88 [14,76; 21,19] | 26,1 | 0,753 |
| Oráculo por par | 17,05 [13,96; 20,25] | 26,3 | 0,752 |

Por clase de fondo (MAE · SSIM):

| Fondo | Sin limpiar | Relleno local | Inpainting Telea | Inpainting NS | Adaptativo (selector actual) | Adaptativo (clase real) |
|---|---|---|---|---|---|---|
| Uniforme | 32,19 · 0,535 | 0,24 · 0,986 | 0,32 · 0,984 | 0,37 · 0,983 | 0,24 · 0,986 | 0,24 · 0,986 |
| Estructurado | 30,23 · 0,691 | 19,08 · 0,636 | 19,13 · 0,653 | 19,05 · 0,654 | 19,01 · 0,651 | 19,13 · 0,653 |
| Complejo | 29,71 · 0,674 | 33,48 · 0,608 | 34,26 · 0,621 | 34,12 · 0,623 | 34,19 · 0,619 | 34,26 · 0,621 |

**Hipótesis** (adaptativo con el selector actual vs. mejor método único):

- MAE: mejor método único = Relleno local; adaptativo − Relleno local = 0,21 [-0,41; 1,03] → **sin diferencia significativa**.
- PSNR: mejor método único = Relleno local; adaptativo − Relleno local = 0,2 [-0,0; 0,5] → **sin diferencia significativa**.
- SSIM: mejor método único = Inpainting NS; adaptativo − Inpainting NS = -0,001 [-0,003; 0,000] → **sin diferencia significativa**.

En la aplicación, el modo automático deja para revisión manual (sin limpiar) 0 de 50 pares uniformes, 9 de 50 pares estructurados, 37 de 50 pares complejos.

## 4. Conservación

Píxeles modificados fuera de la máscara, en todas las limpiezas: **máximo 0** (meta: 0). Dentro de la máscara del sistema, los métodos alteraron de forma visible (más de 8 niveles) en promedio 624,8 píxeles de fondo sano por par: es el costo de que la máscara marque algo más que el texto.

## 5. Detección de texto (OpenMantra)

Detector actual: `detect_bubbles` (heurístico, busca globos ovalados). Una caja de texto está acertada si al menos el 80 % queda dentro de una región detectada; una detección es correcta si contiene así algún texto. Como referencia se incluye el F1 con el criterio estándar (emparejamiento uno a uno con IoU ≥ 0,5), que castiga que el globo sea mucho más grande que el texto.

| Obra | Páginas | Textos | Detecciones | Precisión | Recall | F1 | F1 (IoU ≥ 0,5) |
|---|---|---|---|---|---|---|---|
| tojime_no_siora | 46 | 334 | 533 | 0,45 | 0,72 | **0,55** | 0,22 |
| balloon_dream | 38 | 314 | 175 | 0,58 | 0,40 | **0,48** | 0,14 |
| tencho_isoro (prueba) | 40 | 311 | 206 | 0,83 | 0,58 | **0,68** | 0,20 |
| boureisougi | 36 | 274 | 63 | 0,79 | 0,21 | **0,33** | 0,15 |
| rasetugari | 54 | 359 | 400 | 0,52 | 0,62 | **0,57** | 0,23 |
| **Todas** | 214 | 1592 | 1377 | 0,56 | 0,52 | **0,54** | 0,20 |

## 6. Rendimiento

20 páginas de OpenMantra repartidas entre las cinco obras, ampliadas a 2048 px de lado mayor. Se mide detectar regiones, crear sus máscaras y limpiarlas en modo automático (sin contar abrir el archivo).

| Mediana (s) | Máximo (s) | Regiones por página (media) | Meta |
|---|---|---|---|
| 0,37 | 1,20 | 7,8 | ≤ 15 s |

Equipo: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz, 8 hilos, Windows 10.

## 7. Selector con regiones reales (OpenMantra, etiquetadas a mano)

Casi todo el texto de OpenMantra está en globos blancos, así que al etiquetar se muestran más seguido las regiones con fondo cargado (muestreo por estratos, ver `muestreo.py`). La tabla muestra los conteos reales de regiones etiquetadas; el F1, la exactitud y el macro-F1 están **ponderados** para estimar el resultado en todo OpenMantra.

| Real \ Predicha | Uniforme | Estructurado | Complejo | F1 (ponderado) |
|---|---|---|---|---|
| Uniforme | 315 | 5 | 0 | 0,99 |
| Estructurado | 5 | 6 | 0 | 0,20 |
| Complejo | 3 | 11 | 12 | 0,65 |

357 regiones, etiquetadas por 1 persona(s). Exactitud ponderada: **0,97** · Macro-F1 ponderado: **0,61** (sin ponderar: 0,66). Aún hay pocas regiones de: Estructurado (meta: 15 de cada una).

Los umbrales del selector se ajustaron con estas mismas regiones (salvo las de la obra de prueba tencho_isoro, que nunca se usan para ajustar) y con pares sintéticos de desarrollo (ver `calibrar_selector.py`), así que los valores de arriba son optimistas. La estimación honesta es la **validación cruzada por obra**: para cada obra se ajustan los umbrales sin ella y se mide en ella. Macro-F1 ponderado: **0,61** · F1 por clase: Uniforme 0,99 · Estructurado 0,21 · Complejo 0,65. Umbrales de cada ajuste (entropía en bits · tono oscuro): sin balloon_dream 1,5 · 40 · sin boureisougi 1,5 · 40 · sin rasetugari 1,6 · 40 · sin tencho_isoro 1,5 · 40 · sin tojime_no_siora 1,6 · 40.

| Estrato | Etiquetadas | Regiones en OpenMantra |
|---|---|---|
| 1 | 79 | 81 |
| 2 | 123 | 238 |
| 3 | 63 | 477 |
| 4 | 92 | 796 |
