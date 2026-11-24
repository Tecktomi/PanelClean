# Resultados de la evaluación de PanelClean

Generado con `python -m evaluacion.evaluar` · PanelClean 0.3.1-beta · Python 3.15.2 · OpenCV 4.14.0 · NumPy 2.5.5 · 150 pares sintéticos (semilla 2026) · 214 páginas de OpenMantra.

Los intervalos entre corchetes son de confianza del 95 % (bootstrap). Los datos de cada par y cada página están en los CSV de esta carpeta; `ejemplos.png` muestra algunos pares.

## Resumen frente a las metas del Hito 1

| Métrica | Meta | Resultado | ¿Cumple? |
|---|---|---|---|
| Detección en OpenMantra (F1, criterio actual: centro y tamaño) | ≥ 0,85 | 0,92 | Sí |
| Detección (F1, criterio actual, obra reservada tencho_isoro, consultada durante el desarrollo) | ≥ 0,85 | 0,89 | Sí |
| Detección en OpenMantra (F1, criterio original: cubrir 80 % de la caja) | ≥ 0,85 | 0,13 | No |
| Máscara (IoU medio, sintéticos) | ≥ 0,80 | 0,67 | No |
| Selector (macro-F1, sintéticos) | ≥ 0,80 | 0,86 | Sí |
| Selector (macro-F1, regiones reales, validación cruzada por obra) | ≥ 0,80 | 0,62 | No |
| «Ajustar al texto»: zonas con ≥ 95 % de las letras cubiertas (rectángulo tal cual → ajustado) | sin meta | 77 % → 94 % | — |
| «Ajustar al texto»: dibujo borrado por zona (rectángulo tal cual → ajustado) | sin meta | 272 px → 45 px | — |
| Reconstrucción (MAE/PSNR/SSIM) | sin meta | ver sección 3 | — |
| Conservación (píxeles fuera de la máscara) | 0 | 0 | Sí |
| Rendimiento (máximo por página de 2048 px) | ≤ 15 s | 0,65 s | Sí |
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

Detector de «Detectar texto» y de «Procesar todas»: bloques de tinta rodeados de papel en la página entera, buscados con varios tamaños de letra, filtrados por un random forest de OpenCV que mira 17 medidas de cada bloque (`entrenar_detector.py`); entre los que se solapan queda el más completo. Se informan las regiones con confianza ≥ 0,5, las que el modo automático limpia. Cada obra de desarrollo se mide con un modelo entrenado sin ella. **Criterio**: una región acierta un texto si su centro cae en la caja anotada y no mide más de 3 veces esa caja. **Enteros**: textos con al menos el 85 % de sus letras dentro de lo que se propone; es más exigente, porque con el criterio también acierta un trozo de texto.

**Qué no dice esta tabla.** `tencho_isoro` nunca se usó para entrenar, pero se consultó varias veces durante el desarrollo (para mirar errores y comparar alternativas), así que su resultado no es una prueba a ciegas. Y las cinco obras son del mismo conjunto, todas en japonés: no dice cómo le va en otros idiomas o estilos. Para eso están las series propias que se marcan con `marcar.py`.

| Obra | Textos | Anterior (`detect_bubbles`): P · R · F1 | Precisión | Recall | F1 | Enteros | F1 con el criterio original (anterior → actual) |
|---|---|---|---|---|---|---|---|
| tojime_no_siora | 334 | 0,36 · 0,58 · 0,45 | 0,92 | 0,94 | **0,93** | 0,92 | 0,55 → 0,22 |
| balloon_dream | 314 | 0,31 · 0,17 · 0,22 | 0,94 | 0,94 | **0,94** | 0,91 | 0,48 → 0,16 |
| tencho_isoro (reservada) | 311 | 0,55 · 0,37 · 0,44 | 0,86 | 0,92 | **0,89** | 0,91 | 0,68 → 0,12 |
| boureisougi | 274 | 0,65 · 0,15 · 0,24 | 0,95 | 0,92 | **0,93** | 0,75 | 0,33 → 0,03 |
| rasetugari | 359 | 0,39 · 0,43 · 0,41 | 0,90 | 0,93 | **0,92** | 0,83 | 0,57 → 0,09 |
| **Desarrollo** | 1281 | 0,38 · 0,35 · 0,36 | 0,92 | 0,93 | **0,93** | 0,86 | 0,51 → 0,13 |
| **Todas** | 1592 | 0,41 · 0,35 · 0,38 | 0,91 | 0,93 | **0,92** | 0,87 | 0,54 → 0,13 |

El criterio original, usado hasta el 23/11, pedía cubrir el 80 % de la caja anotada. Las cajas de OpenMantra son más holgadas que las letras y las regiones actuales se ajustan a ellas, así que casi nunca lo cumplen aunque contengan todo el texto. Se informan los dos: el cambio de criterio favorece al detector actual.

Contando también las dudosas (confianza ≥ 0,3, en naranja, que no se limpian solas), el recall es 0,95 y la precisión 0,81. El techo, si se aceptaran todos los candidatos, es 1,00.

**Dibujo que borraría el modo automático.** De los píxeles que se limpian en todas las páginas, 5,3 % están fuera de las cajas de texto anotadas: 141.956 de 2.681.833. Con `detect_bubbles` eran 16 % (172.113 de 1.056.947).

**Texto horizontal simulado.** En las páginas de desarrollo con los globos traducidos al inglés (`horizontales.py`), cada obra sin verla: precisión 0,90, recall 0,84 y techo 0,95. Como escaneo chico y comprimido (500 a 620 px de ancho): precisión 0,89, recall 0,80 y techo 0,93. Es una aproximación: las fuentes son las mismas del entrenamiento y no hay páginas reales.

**Escaneos distintos y clics** (30 páginas de desarrollo; el modelo, que vio esas páginas sin alterar, solo elige entre bloques que se solapan). «Candidatos»: textos que el detector llega a proponer. «Clic»: textos que quedan con una región correcta al hacer clic en su centro. «Gigantes»: clics que confirman una región de más del 15 % de la hoja.

| Variante | Textos | Candidatos | Clic | Gigantes |
|---|---|---|---|---|
| Original | 195 | 95 % | 95 % | 0 |
| Papel 18 % más oscuro | 195 | 95 % | 95 % | 0 |
| Papel amarillento | 195 | 95 % | 95 % | 0 |
| Tinta lavada | 195 | 95 % | 95 % | 0 |
| Sombra hacia un lado | 195 | 52 % | 96 % | 0 |
| Resolución × 2 | 195 | 96 % | 96 % | 0 |
| Resolución × 0,6 | 195 | 96 % | 96 % | 0 |

**Series propias** (marcadas con `marcar.py`). «Sin verla»: el modelo no se entrenó con esa serie. «Viendo otras de sus páginas»: se entrenó con parte de ellas y se midió en las demás. Si la serie no es reservada, sus cajas partieron de las propuestas del detector, y eso lo favorece algo. Además la serie 1 se usó para encontrar qué fallaba y elegir cómo arreglarlo, así que **su resultado es optimista**: para uno a ciegas hace falta una serie reservada.

| Serie | Textos | Candidatos (techo) | Precisión | Recall | F1 | Enteros |
|---|---|---|---|---|---|---|
| Serie 1 (sin verla) | 271 | 0,93 | 0,97 | 0,64 | **0,77** | 0,60 |
| Serie 1 (viendo otras de sus páginas) | 271 | 0,93 | 0,96 | 0,74 | **0,83** | 0,73 |

## 6. Rendimiento

20 páginas de OpenMantra repartidas entre las cinco obras, ampliadas a 2048 px de lado mayor. Se mide detectar regiones, crear sus máscaras y limpiarlas en modo automático (sin contar abrir el archivo).

| Mediana (s) | Máximo (s) | Regiones por página (media) | Meta |
|---|---|---|---|
| 0,42 | 0,65 | 7,8 | ≤ 15 s |

**Páginas más grandes** (¿conviene reducirlas a 2048 px antes de procesar, T1.2.4?). Tiempo máximo: 3500 px, 0,86 s (5 páginas) · 5000 px, 1,26 s (5 páginas). No conviene: la búsqueda de texto ya se hace a un tamaño fijo (830 px de ancho), el tiempo está lejos de la meta, y reducir la página cambiaría todos sus píxeles (la meta es no cambiar ninguno fuera de la máscara) y la exportaría con menos resolución que la original.

Equipo: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz, 8 hilos, Windows 10.

## 7. Selector con regiones reales (OpenMantra, etiquetadas a mano)

Casi todo el texto de OpenMantra está en globos blancos, así que al etiquetar se muestran más seguido las regiones con fondo cargado (muestreo por estratos, ver `muestreo.py`). La tabla muestra los conteos reales de regiones etiquetadas; el F1, la exactitud y el macro-F1 están **ponderados** para estimar el resultado en todo OpenMantra.

| Real \ Predicha | Uniforme | Estructurado | Complejo | F1 (ponderado) |
|---|---|---|---|---|
| Uniforme | 316 | 3 | 1 | 0,99 |
| Estructurado | 1 | 4 | 0 | 0,26 |
| Complejo | 2 | 11 | 12 | 0,65 |

350 regiones, etiquetadas por 1 persona(s). Exactitud ponderada: **0,98** · Macro-F1 ponderado: **0,63** (sin ponderar: 0,66). Aún hay pocas regiones de: Estructurado (meta: 15 de cada una).

Los umbrales del selector se ajustaron con estas mismas regiones (salvo las de la obra de prueba tencho_isoro, que nunca se usan para ajustar) y con pares sintéticos de desarrollo (ver `calibrar_selector.py`), así que los valores de arriba son optimistas. La estimación honesta es la **validación cruzada por obra**: para cada obra se ajustan los umbrales sin ella y se mide en ella. Macro-F1 ponderado: **0,62** · F1 por clase: Uniforme 1,00 · Estructurado 0,24 · Complejo 0,63. Umbrales de cada ajuste (entropía en bits · tono oscuro): sin balloon_dream 2,5 · 40 · sin boureisougi 2,5 · 100 · sin rasetugari 2,0 · 50 · sin tencho_isoro 2,5 · 80 · sin tojime_no_siora 2,5 · 80.

| Estrato | Etiquetadas | Regiones en OpenMantra |
|---|---|---|
| 1 | 76 | 81 |
| 2 | 121 | 238 |
| 3 | 62 | 477 |
| 4 | 91 | 796 |

271 regiones se etiquetaron dos veces por la misma persona. Las etiquetas coinciden en 262 (97 %; kappa de Cohen 0,86). Coincidencia por clase (F1 entre las dos): Uniforme 0,98 · Estructurado 0,53 · Complejo 0,98. Las 9 que no coinciden quedan fuera. **Es consistencia de una persona consigo misma, no acuerdo entre personas**: no se consiguió que otra etiquetara, así que la referencia depende de un solo criterio. Donde menos coincide es en «Estructurado»: ahí la referencia misma es inestable, y eso limita lo que se puede medir del selector en esa clase.

## 8. Selección por zona aproximada («Ajustar al texto», OpenMantra)

Simula a una persona que encierra cada texto a grandes rasgos: la caja anotada más un margen al azar de 10–70 % de su lado corto en cada lado (más 6 px). Se compara usar ese rectángulo tal cual como región con buscar el texto dentro de él (`texto_en_zona`); cada celda dice «tal cual → **ajustado**». Las letras de referencia son la máscara que da la caja anotada; el dibujo borrado son los píxeles de máscara fuera de todas las cajas anotadas. Los parámetros se ajustaron con las cuatro obras de desarrollo; la reservada se consultó al desarrollarlo (para ver sus errores y probar una regla). Si no encuentra texto, la aplicación usa el rectángulo tal cual, y aquí también.

| Obra | Zonas | Letras cubiertas (media) | Zonas con ≥ 95 % cubierto | Dibujo borrado (px por zona) | Zonas que borran > 50 px de dibujo | Bloques por texto | Bloques fuera de las cajas |
|---|---|---|---|---|---|---|---|
| tojime_no_siora | 120 | 0,97 → **0,99** | 81 % → **96 %** | 217 → **29** | 53 % → **7 %** | 1,18 | 8 % |
| balloon_dream | 120 | 0,97 → **0,99** | 87 % → **96 %** | 415 → **73** | 54 % → **8 %** | 1,27 | 5 % |
| tencho_isoro (reservada) | 120 | 0,98 → **0,99** | 88 % → **97 %** | 314 → **85** | 66 % → **23 %** | 1,38 | 22 % |
| boureisougi | 120 | 0,86 → **0,96** | 49 % → **88 %** | 219 → **25** | 54 % → **6 %** | 1,15 | 4 % |
| rasetugari | 120 | 0,95 → **0,99** | 80 % → **96 %** | 197 → **14** | 53 % → **7 %** | 1,09 | 4 % |
| **Desarrollo (sin la reservada)** | 480 | 0,94 → **0,98** | 74 % → **94 %** | 262 → **35** | 54 % → **7 %** | 1,17 | 5 % |
| **Todas** | 600 | 0,95 → **0,98** | 77 % → **94 %** | 272 → **45** | 56 % → **10 %** | 1,21 | 9 % |

En 1 % de las zonas no encontró texto en globos (se usa el rectángulo). Un bloque «fuera de las cajas» tiene menos de la mitad de su área dentro de una caja anotada: casi siempre son trozos de dibujo encerrados por papel blanco (trazos gruesos, líneas punteadas que decoran el borde del globo), y a veces texto que OpenMantra no anotó.
