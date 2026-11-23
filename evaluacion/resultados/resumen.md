# Resultados de la evaluación de PanelClean

Generado con `python -m evaluacion.evaluar` · PanelClean 0.3.1-beta · Python 3.15.2 · OpenCV 4.14.0 · NumPy 2.5.5 · 150 pares sintéticos (semilla 2026) · 214 páginas de OpenMantra.

Los intervalos entre corchetes son de confianza del 95 % (bootstrap). Los datos de cada par y cada página están en los CSV de esta carpeta; `ejemplos.png` muestra algunos pares.

## Resumen frente a las metas del Hito 1

| Métrica | Meta | Resultado | ¿Cumple? |
|---|---|---|---|
| Detección (F1, todas las obras) | ≥ 0,85 | 0,54 | No |
| Detección (F1, obra de prueba tencho_isoro) | ≥ 0,85 | 0,68 | No |
| Máscara (IoU medio, sintéticos) | ≥ 0,80 | 0,67 | No |
| Selector (macro-F1, sintéticos) | ≥ 0,80 | 0,86 | Sí |
| Selector (macro-F1, regiones reales, validación cruzada por obra) | ≥ 0,80 | 0,57 | No |
| «Ajustar al texto»: zonas con ≥ 95 % de las letras cubiertas (rectángulo tal cual → ajustado) | sin meta | 77 % → 91 % | — |
| «Ajustar al texto»: dibujo borrado por zona (rectángulo tal cual → ajustado) | sin meta | 272 px → 41 px | — |
| Reconstrucción (MAE/PSNR/SSIM) | sin meta | ver sección 3 | — |
| Conservación (píxeles fuera de la máscara) | 0 | 0 | Sí |
| Rendimiento (máximo por página de 2048 px) | ≤ 15 s | 0,98 s | Sí |
| Hipótesis: adaptativo vs. mejor método único (máscara perfecta) | adaptativo mejor | MAE: peor · PSNR: mejor · SSIM: peor | — |
| Hipótesis: adaptativo vs. mejor método único (máscara del sistema) | adaptativo mejor | MAE: sin diferencia significativa · PSNR: mejor · SSIM: sin diferencia significativa | — |

## 1. Máscara de texto (pares sintéticos)

Referencia: los píxeles que el texto cambió respecto del fondo limpio (diferencia mayor que 8 niveles de gris), borde blanco incluido. La máscara se genera con los ajustes por defecto de la aplicación. *Cobertura* = parte del texto que queda dentro de la máscara (lo que falta queda como resto); *exceso* = parte de la máscara que no era texto.

| Fondo | Pares | IoU [IC 95 %] | Dice | Cobertura | Exceso |
|---|---|---|---|---|---|
| Uniforme | 50 | 0,98 [0,98; 0,98] | 0,99 | 0,99 | 0,01 |
| Estructurado | 50 | 0,65 [0,60; 0,70] | 0,77 | 0,81 | 0,18 |
| Complejo | 50 | 0,37 [0,30; 0,43] | 0,48 | 0,43 | 0,26 |
| Con borde blanco | 44 | 0,56 [0,52; 0,59] | 0,71 | 0,62 | 0,14 |
| Sin borde blanco | 106 | 0,71 [0,65; 0,78] | 0,76 | 0,79 | 0,16 |
| **Todos** | 150 | 0,67 [0,61; 0,71] | 0,75 | 0,74 | 0,15 |

## 2. Selector de estrategia (pares sintéticos)

La clase real es la del fondo generado. Se evalúa la regla de entropía y tono oscuro (`classify_background`) sin el aviso de «Revisar máscara», que no es una clase de fondo. Sus umbrales se ajustaron con otros pares (semilla 7, ver `calibrar_selector.py`), así que estos pares no se usaron para ajustarla.

| Real \ Predicha | Uniforme | Estructurado | Complejo | F1 |
|---|---|---|---|---|
| Uniforme | 50 | 0 | 0 | 0,91 |
| Estructurado | 5 | 43 | 2 | 0,85 |
| Complejo | 5 | 8 | 37 | 0,83 |

Exactitud: **0,87** · Macro-F1: **0,86** · Regiones con aviso «Revisar máscara» en la aplicación: 7 de 150.

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
| Adaptativo (selector actual) | 6,55 [5,57; 7,56] | 31,2 | 0,900 |
| Adaptativo (clase real) | 6,51 [5,53; 7,53] | 31,3 | 0,901 |
| Oráculo por par | 5,97 [5,05; 6,93] | 31,2 | 0,899 |

Por clase de fondo (MAE · SSIM):

| Fondo | Sin limpiar | Relleno local | Inpainting Telea | Inpainting NS | Adaptativo (selector actual) | Adaptativo (clase real) |
|---|---|---|---|---|---|---|
| Uniforme | 32,19 · 0,535 | 0,21 · 0,992 | 0,28 · 0,989 | 0,29 · 0,990 | 0,21 · 0,992 | 0,21 · 0,992 |
| Estructurado | 30,23 · 0,691 | 8,88 · 0,813 | 9,83 · 0,840 | 9,68 · 0,843 | 9,83 · 0,840 | 9,83 · 0,840 |
| Complejo | 29,71 · 0,674 | 16,64 · 0,793 | 9,49 · 0,871 | 9,22 · 0,873 | 9,62 · 0,869 | 9,49 · 0,871 |

**Hipótesis** (adaptativo con el selector actual vs. mejor método único):

- MAE: mejor método único = Inpainting NS; adaptativo − Inpainting NS = 0,16 [0,08; 0,26] → **peor**.
- PSNR: mejor método único = Inpainting Telea; adaptativo − Inpainting Telea = 0,4 [0,2; 0,6] → **mejor**.
- SSIM: mejor método único = Inpainting NS; adaptativo − Inpainting NS = -0,002 [-0,003; -0,000] → **peor**.

En la aplicación, el modo automático deja para revisión manual (sin limpiar) 0 de 50 pares uniformes, 5 de 50 pares estructurados, 38 de 50 pares complejos.

### Con la máscara del sistema (resultado que obtiene el usuario)

| Estrategia | MAE [IC 95 %] | PSNR (dB) | SSIM |
|---|---|---|---|
| Sin limpiar | 30,71 [29,27; 32,21] | 10,8 | 0,633 |
| Relleno local | 13,68 [11,47; 15,90] | 26,8 | 0,794 |
| Inpainting Telea | 13,84 [11,70; 16,05] | 26,6 | 0,803 |
| Inpainting NS | 13,85 [11,71; 16,04] | 26,2 | 0,804 |
| Adaptativo (selector actual) | 13,51 [11,39; 15,67] | 27,1 | 0,804 |
| Adaptativo (clase real) | 13,81 [11,68; 16,03] | 27,0 | 0,804 |
| Oráculo por par | 13,13 [11,07; 15,23] | 27,2 | 0,802 |

Por clase de fondo (MAE · SSIM):

| Fondo | Sin limpiar | Relleno local | Inpainting Telea | Inpainting NS | Adaptativo (selector actual) | Adaptativo (clase real) |
|---|---|---|---|---|---|---|
| Uniforme | 32,19 · 0,535 | 0,23 · 0,989 | 0,31 · 0,986 | 0,35 · 0,986 | 0,23 · 0,989 | 0,23 · 0,989 |
| Estructurado | 30,23 · 0,691 | 15,88 · 0,695 | 15,98 · 0,712 | 15,96 · 0,714 | 15,88 · 0,710 | 15,98 · 0,712 |
| Complejo | 29,71 · 0,674 | 24,93 · 0,699 | 25,23 · 0,712 | 25,25 · 0,712 | 24,41 · 0,713 | 25,23 · 0,712 |

**Hipótesis** (adaptativo con el selector actual vs. mejor método único):

- MAE: mejor método único = Relleno local; adaptativo − Relleno local = -0,17 [-0,62; 0,17] → **sin diferencia significativa**.
- PSNR: mejor método único = Relleno local; adaptativo − Relleno local = 0,3 [0,1; 0,5] → **mejor**.
- SSIM: mejor método único = Inpainting NS; adaptativo − Inpainting NS = -0,000 [-0,002; 0,002] → **sin diferencia significativa**.

En la aplicación, el modo automático deja para revisión manual (sin limpiar) 0 de 50 pares uniformes, 5 de 50 pares estructurados, 37 de 50 pares complejos.

## 4. Conservación

Píxeles modificados fuera de la máscara, en todas las limpiezas: **máximo 0** (meta: 0). Dentro de la máscara del sistema, los métodos alteraron de forma visible (más de 8 niveles) en promedio 359,4 píxeles de fondo sano por par: es el costo de que la máscara marque algo más que el texto.

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

**Modo automático después de «Detectar globos».** Si se limpian todas las regiones detectadas sin revisarlas, el modo automático limpia 1124 y deja 253 para revisión. De los píxeles que borra, 172.113 (16 %) están fuera de las cajas de texto anotadas: es dibujo que se pierde. Las regiones de más del 5 % de la página (y más de 40.000 px) quedan para revisión, porque casi nunca son solo texto: suelen ser viñetas enteras.

## 6. Rendimiento

20 páginas de OpenMantra repartidas entre las cinco obras, ampliadas a 2048 px de lado mayor. Se mide detectar regiones, crear sus máscaras y limpiarlas en modo automático (sin contar abrir el archivo).

| Mediana (s) | Máximo (s) | Regiones por página (media) | Meta |
|---|---|---|---|
| 0,40 | 0,98 | 7,8 | ≤ 15 s |

Equipo: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz, 8 hilos, Windows 10.

## 7. Selector con regiones reales (OpenMantra, etiquetadas a mano)

Casi todo el texto de OpenMantra está en globos blancos, así que al etiquetar se muestran más seguido las regiones con fondo cargado (muestreo por estratos, ver `muestreo.py`). La tabla muestra los conteos reales de regiones etiquetadas; el F1, la exactitud y el macro-F1 están **ponderados** para estimar el resultado en todo OpenMantra.

| Real \ Predicha | Uniforme | Estructurado | Complejo | F1 (ponderado) |
|---|---|---|---|---|
| Uniforme | 316 | 3 | 1 | 0,99 |
| Estructurado | 4 | 4 | 0 | 0,17 |
| Complejo | 2 | 11 | 12 | 0,65 |

353 regiones, etiquetadas por 2 persona(s). Exactitud ponderada: **0,97** · Macro-F1 ponderado: **0,60** (sin ponderar: 0,64). Aún hay pocas regiones de: Estructurado (meta: 15 de cada una).

Los umbrales del selector se ajustaron con estas mismas regiones (salvo las de la obra de prueba tencho_isoro, que nunca se usan para ajustar) y con pares sintéticos de desarrollo (ver `calibrar_selector.py`), así que los valores de arriba son optimistas. La estimación honesta es la **validación cruzada por obra**: para cada obra se ajustan los umbrales sin ella y se mide en ella. Macro-F1 ponderado: **0,57** · F1 por clase: Uniforme 0,99 · Estructurado 0,16 · Complejo 0,55. Umbrales de cada ajuste (entropía en bits · tono oscuro): sin balloon_dream 2,0 · 40 · sin boureisougi 2,0 · 40 · sin rasetugari 2,0 · 50 · sin tencho_isoro 2,0 · 40 · sin tojime_no_siora 1,7 · 40.

| Estrato | Etiquetadas | Regiones en OpenMantra |
|---|---|---|
| 1 | 76 | 81 |
| 2 | 122 | 238 |
| 3 | 63 | 477 |
| 4 | 92 | 796 |

134 regiones tienen etiquetas de más de una persona; coinciden en 129 (96 %). Las que no coinciden quedan fuera hasta resolverlas.

## 8. Selección por zona aproximada («Ajustar al texto», OpenMantra)

Simula a una persona que encierra cada texto a grandes rasgos: la caja anotada más un margen al azar de 10–70 % de su lado corto en cada lado (más 6 px). Se compara usar ese rectángulo tal cual como región con buscar el texto dentro de él (`texto_en_zona`); cada celda dice «tal cual → **ajustado**». Las letras de referencia son la máscara que da la caja anotada; el dibujo borrado son los píxeles de máscara fuera de todas las cajas anotadas. Los parámetros se ajustaron solo con las cuatro obras de desarrollo. Si no encuentra texto, la aplicación usa el rectángulo tal cual, y aquí también.

| Obra | Zonas | Letras cubiertas (media) | Zonas con ≥ 95 % cubierto | Dibujo borrado (px por zona) | Zonas que borran > 50 px de dibujo | Bloques por texto | Bloques fuera de las cajas |
|---|---|---|---|---|---|---|---|
| tojime_no_siora | 120 | 0,97 → **0,98** | 81 % → **93 %** | 217 → **24** | 53 % → **6 %** | 1,19 | 8 % |
| balloon_dream | 120 | 0,97 → **0,97** | 87 % → **93 %** | 415 → **69** | 54 % → **8 %** | 1,26 | 5 % |
| tencho_isoro (prueba) | 120 | 0,98 → **0,98** | 88 % → **95 %** | 314 → **73** | 66 % → **22 %** | 1,37 | 22 % |
| boureisougi | 120 | 0,86 → **0,95** | 49 % → **85 %** | 219 → **22** | 54 % → **6 %** | 1,14 | 4 % |
| rasetugari | 120 | 0,95 → **0,97** | 80 % → **90 %** | 197 → **16** | 53 % → **7 %** | 1,09 | 4 % |
| **Desarrollo (sin la prueba)** | 480 | 0,94 → **0,97** | 74 % → **90 %** | 262 → **33** | 54 % → **7 %** | 1,17 | 5 % |
| **Todas** | 600 | 0,95 → **0,97** | 77 % → **91 %** | 272 → **41** | 56 % → **10 %** | 1,21 | 9 % |

En 1 % de las zonas no encontró texto en globos (se usa el rectángulo). Un bloque «fuera de las cajas» tiene menos de la mitad de su área dentro de una caja anotada: casi siempre son trozos de dibujo encerrados por papel blanco (trazos gruesos, líneas punteadas que decoran el borde del globo), y a veces texto que OpenMantra no anotó.
