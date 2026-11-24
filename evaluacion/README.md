# Evaluación de PanelClean

Mide, de forma reproducible, las metas comprometidas en el Hito 1 (presentación, diapositiva 7 y respaldo B). Los resultados más recientes están en [`resultados/resumen.md`](resultados/resumen.md).

## Cómo correrla

Desde la raíz del repositorio, con el entorno ya instalado:

```bash
.venv\Scripts\python -m evaluacion.evaluar      # Windows
.venv/bin/python -m evaluacion.evaluar          # Linux
```

Tarda unos 10 minutos en un portátil, porque vuelve a medir los candidatos de OpenMantra. Con `--rapido` corre una versión corta para probar. Repetirla da los mismos resultados, salvo los tiempos. No necesita instalar nada más que lo de `requirements.txt`.

## Qué se mide

| # | Métrica | Tarea | Datos | Definición | Meta (Hito 1) |
|---|---|---|---|---|---|
| 1 | Detección: precisión, recall, F1 | T1.2.3, T1.2.6 | OpenMantra, 214 páginas; series propias marcadas a mano | Criterio actual: una región acierta un texto si su centro cae en la caja anotada y no mide más de 3 veces esa caja. Se informa también el criterio original (cubrir el 80 % de la caja), el detector anterior (`detect_bubbles`) con los dos, cuánto dibujo borraría el modo automático, y qué pasa con escaneos distintos y con los clics del usuario | F1 ≥ 0,85 |
| 2 | Máscara: IoU, Dice, cobertura, exceso | T1.4.2 | 150 pares sintéticos | Contra los píxeles que el texto cambió respecto del fondo limpio, con los ajustes por defecto de la aplicación | IoU ≥ 0,80 |
| 3 | Selector: exactitud, macro-F1, matriz de confusión | T2.4.4 | Sintéticos (clase por construcción) y regiones reales etiquetadas | Regla de entropía, tono oscuro y textura (`classify_background`), sin el aviso «Revisar máscara». En reales se informa la validación cruzada por obra | Macro-F1 ≥ 0,80 |
| 4 | Limpieza: MAE, PSNR, SSIM | T2.4.1 | Sintéticos | Salida vs. fondo limpio, en gris, dentro de la selección | Sin meta |
| 5 | Conservación | T2.4.2 | Sintéticos | Píxeles cambiados fuera de la máscara; además, fondo sano alterado dentro de ella | 0 píxeles |
| 6 | Rendimiento | T2.4.3 | 20 páginas de OpenMantra ampliadas a 2048 px | Tiempo de detectar, crear máscaras y limpiar en automático | ≤ 15 s/página |
| 7 | Hipótesis: adaptativo vs. método único | T2.4.5 | Sintéticos | Diferencia pareada con el mejor método único, IC 95 % por bootstrap, en MAE, PSNR y SSIM | Adaptativo mejor |
| 8 | «Ajustar al texto»: letras cubiertas y dibujo borrado | T1.2.3 | OpenMantra, 120 zonas por obra | Zonas holgadas alrededor de cada texto (caja anotada + 10–70 % de su lado corto por lado). Se compara usar la zona tal cual con buscar el texto dentro de ella: qué parte de las letras cubre la máscara y cuántos píxeles borra fuera de las cajas anotadas | Sin meta |

## Datos

**Pares sintéticos** (`sinteticos.py`, tarea T1.4.1). Fondo limpio + texto japonés vertical. La clase del fondo se conoce por construcción:

- *Uniforme*: blanco o gris liso con ruido de escaneo.
- *Estructurado*: trama de puntos, rayado paralelo o degradado.
- *Complejo*: trazos irregulares, manchas oscuras y parches de trama que imitan dibujo.

Son 50 pares por clase, siempre los mismos (semilla 2026). Las frases salen de las anotaciones de OpenMantra y la fuente es Noto Sans JP Bold. Una parte del texto lleva contorno blanco, como el que se usa sobre dibujo. La **referencia** son los píxeles cuya diferencia con el fondo limpio supera 8 niveles de gris; la **selección** es el rectángulo del texto más 10 px, como un rectángulo ajustado hecho por el usuario.

**OpenMantra** (`datasets/open-mantra`). Detección, rendimiento y selector con regiones reales. La obra `tencho_isoro` está reservada: no se entrena ni se calculan umbrales con ella. Pero **no es una prueba a ciegas**: durante el desarrollo del detector y de «Ajustar al texto» se consultó varias veces, para mirar errores y comparar alternativas (por ejemplo, con ella se descartó usar el buscador en toda la página sin filtro). Además es del mismo conjunto que las demás obras, en japonés.

**Series propias** (`cajas_texto.json`). Páginas de otros estilos o idiomas marcadas con `marcar.py`. Las que se marcan como reservadas son la única prueba a ciegas del detector: no se usan para entrenar ni para elegir nada.

## Decisiones de diseño

- **Máscara del sistema y máscara perfecta.** La limpieza se mide dos veces. Con la máscara perfecta se aísla la elección del método; con la del sistema se ve lo que obtiene el usuario.
- **Tres métricas para la hipótesis.** El MAE favorece por construcción el relleno con la mediana (la mediana minimiza el error absoluto) aunque borre la textura; el SSIM sí mide si la textura se conserva.
- **Adaptativo comparable en todos los pares.** La aplicación deja los fondos complejos para revisión manual. Para comparar en los mismos pares, el adaptativo usa Relleno local si el fondo se clasifica como Uniforme y Telea en otro caso. Aparte se informa cuántos pares deja la aplicación para revisión.
- **Piso y techo.** *Sin limpiar* es el piso; el *oráculo por par* (el mejor método único en cada par) es el techo de cualquier selector.
- **Selector calibrado sin la obra reservada.** La regla mide el fondo en una franja de hasta 14 px alrededor de las letras: la *entropía* del histograma separa el papel liso del resto, y el *tono oscuro* (percentil 10) separa el dibujo, que tiene tinta casi negra, de la trama o el degradado, que no la tienen. Una tercera medida, la *textura repartida* (fracción de la franja que varía en ventanas de 7 × 7 px), evita confundir con papel una trama rala que cubre toda la franja. Los umbrales se calculan con `calibrar_selector.py` a partir de pares sintéticos de desarrollo (semilla 7) y de las regiones reales etiquetadas, sin la obra de prueba. Como las regiones reales sirven para ajustar y para medir, el valor que cuenta es la validación cruzada por obra: se ajusta sin una obra y se mide en ella. La varianza y la densidad de bordes de la regla anterior se descartaron porque en manga real las tramas son finas y de poco contraste: la regla las tomaba por papel liso y, en los sintéticos, tomaba las tramas por dibujo. Con las etiquetas sumadas el 23/11 la calibración propone otros umbrales (2,5 bits y tono 80), pero **no se adoptaron**: en los sintéticos de desarrollo bajan el macro-F1 de 0,92 a 0,77, y su ventaja en las reales (0,67 → 0,82, optimista) depende de 5 regiones estructuradas; en la validación cruzada el macro-F1 sigue en 0,62.
- **Zona aproximada.** `texto_en_zona` toma como letras lo oscuro que el papel blanco de un globo rodea por completo, así el fondo negro o la trama de alrededor no desvían el umbral. Sus parámetros se ajustaron con zonas simuladas de las cuatro obras de desarrollo. La obra reservada se miró más de una vez: con la primera versión, para ver qué eran sus bloques falsos, para probar una regla que se descartó, y con la versión final (el descarte de tramas y contornos de globo salió de la página de ejemplo). Desde el 23/11 la página se prepara antes de buscar (tono de papel y tamaño comunes) y hay topes al tamaño de letra y de bloque. Como las cajas de OpenMantra son holgadas, no sirve medir el IoU contra ellas: la referencia son las letras que da la máscara de la caja anotada.
- **Detector entrenado.** `entrenar_detector.py` toma como candidatos los bloques de texto de cada página, les mide 17 rasgos (grosor y regularidad del trazo, columnas o renglones, papel alrededor, si están en un globo…) y entrena un random forest de OpenCV (`cv2.ml.RTrees`, 30 árboles de profundidad 7, semilla fija). Cada obra de desarrollo se mide con un modelo entrenado sin ella. La obra reservada no se usa para entrenar, pero se midió con varias versiones del detector, así que su resultado es solo orientativo. Las páginas se usan también traspuestas, con los globos traducidos en renglones (`horizontales.py`) y así traducidas pero como un escaneo chico y comprimido; los rasgos no dependen de la orientación, para que el modelo no aprenda solo columnas verticales. Los umbrales (proponer desde 0,3, limpiar desde 0,5) y el tamaño del bosque se eligieron con la validación cruzada.
- **Varios tamaños de letra.** El tamaño de letra se estima en cada zona de papel. Si un globo está abierto, o comparte papel con figuras grandes, el estimado no sirve: el texto se une al dibujo o se descarta. Era la causa de que, en la serie en español, el 26 % del texto no llegara a proponerse (no que estuviera sobre dibujo o en globos oscuros, como se supuso al principio). Ahora se prueban además cuatro tamaños (de 36 a 9 px en una hoja de 830 px). Como salen bloques repetidos y trozos de texto, (1) para entrenar, un bloque es texto si contiene textos enteros (85 % de sus letras) sin medir más de 3 veces lo que miden, y los trozos no se usan; y (2) al detectar, `sin_solapes` deja el bloque más completo si no es mucho menos confiable que sus trozos.
- **Qué se eligió mirando la serie en español.** Los tamaños de letra, el entrenamiento con escaneos chicos, qué cuenta como trozo y cuánto se prefiere el bloque completo se compararon con esa serie (y con OpenMantra, para no empeorarlo). Su medida «sin verla» es honesta respecto del modelo, que no se entrena con ella, pero no respecto de esas decisiones: es optimista. Entre entrenamientos con otra semilla, su F1 varía unas 0,01 a 0,03. También se probó, y se descartó, buscar con otros dos umbrales de papel (180 y 150): llegan a proponerse más textos (96 %), pero el modelo no los distingue mejor del dibujo.
- **Máscara en regiones ajustadas.** Los filtros de la máscara (líneas largas, piezas muy anchas, máscara de más del 36 % de la región) suponían un rectángulo holgado hecho a mano. En una región ajustada a las letras (detectada, o de «Ajustar al texto»), una pieza que queda entera adentro es texto aunque sea tan ancha como una línea, y la máscara puede cubrir hasta el 55 %. Sin esto, en la serie en español quedaban sin limpiar 38 de los 211 textos bien detectados (ahora, 5) y renglones sueltos donde las letras se tocan. Las selecciones a mano no cambian.
- **Dos criterios para la detección.** Hasta el 23/11 una región debía cubrir el 80 % de la caja anotada. Las cajas de OpenMantra son más holgadas que las letras, así que una región ajustada a ellas casi nunca lo cumplía. El criterio actual mira el centro y pone un tope al tamaño, porque con el centro solo una región de media hoja podía contar como acierto. El cambio favorece al detector actual; por eso se informan los dos, y los dos se aplican también al detector anterior.
- **Sin dependencias nuevas.** El SSIM está programado con OpenCV según Wang et al. (2004); las métricas tienen pruebas en `tests/test_evaluacion.py`.

## Etiquetar regiones reales (selector)

La meta del selector se comprometió con regiones reales, así que hay que etiquetar a mano el fondo de algunas cajas de texto de OpenMantra.

En Windows basta con hacer **doble clic en `ETIQUETAR_WINDOWS.bat`** (en la carpeta principal): pide tu nombre y abre la ventana. Si el entorno no está instalado, lo instala primero. También se puede abrir desde la consola:

```bash
.venv\Scripts\python -m evaluacion.etiquetar --nombre TuNombre      # Windows
.venv/bin/python -m evaluacion.etiquetar --nombre TuNombre          # Linux
```

Ojo: hay que usar el Python del entorno (`.venv`), no el `python` del sistema, que no tiene las bibliotecas del proyecto.

Se abre una ventana con una región por vez. La pregunta es **qué hay detrás de las letras, dentro del rectángulo rojo**: es lo que habría que reconstruir al borrarlas, y lo de afuera no cuenta (por eso aparece atenuado). Se responde con el teclado: 1 = Uniforme (papel liso), 2 = Estructurado (trama, rayas o degradado), 3 = Complejo (dibujo), 0 = Descartar; Retroceso deshace. Cada respuesta se guarda al instante en `evaluacion/etiquetas.csv`.

- Meta: al menos 15 regiones de cada clase (la ventana muestra el avance).
- Todas las personas ven las regiones en el mismo orden. Si una región se etiqueta más de una vez, la evaluación informa cuánto coinciden las etiquetas y deja fuera las que no. Las etiquetas actuales son **de una sola persona, en dos pasadas** (guardadas como «tomas» y «tomas ramdohr»): miden su consistencia, no el acuerdo entre personas. No se consiguió que otra persona etiquetara.
- Para revisar las regiones que uno ya marcó como no lisas: `ETIQUETAR_WINDOWS.bat TuNombre --repasar` (o `--repasar` en el comando de arriba).
- Después de etiquetar se vuelve a correr `python -m evaluacion.evaluar`.

**Por qué no salen al azar.** En OpenMantra casi todo el texto está en globos blancos: en 107 regiones mostradas al azar, 103 eran lisas. Para no tener que revisar el dataset entero, `muestreo.py` reparte las 1592 regiones en 4 estratos según cuántos grises intermedios (ni tinta ni papel) hay dentro de la caja, y muestra más seguido las del estrato más cargado. Dentro de cada estrato el orden es al azar. La evaluación pondera cada región por N_h / n_h (regiones del estrato en todo el dataset ÷ regiones etiquetadas de ese estrato), así el resultado estima el desempeño en todo OpenMantra y no solo en las regiones mostradas.

## Marcar páginas propias (detector de texto)

El detector aprendió con OpenMantra, que es manga en japonés. Para enseñarle otros estilos o idiomas, y para saber de verdad cómo le va en ellos, se marca dónde está el texto en páginas propias:

```bash
MARCAR_WINDOWS.bat                                   # Windows (o arrastrar encima la carpeta de una serie)
.venv/bin/python -m evaluacion.marcar ruta/a/la/serie    # Linux
```

- Una carpeta por serie. En cada página, un rectángulo por bloque de texto, ajustado a las letras. El detector propone los suyos para partir de ahí.
- **Todo** el texto de la página: lo que queda sin marcar en una página lista se aprende como «no texto». Las onomatopeyas y letreros que son parte del dibujo se marcan como «ignorar» y no cuentan para ningún lado.
- **Reservar al menos una serie** (casilla de la ventana) antes de marcarla: no se entrena con ella y no muestra propuestas, porque una referencia que parte de lo que encontró el detector lo favorece. Es la medida que vale para otros idiomas.
- Conviene marcar pocas páginas de varias series antes que muchas de una sola.
- **Las imágenes no se copian ni se suben**, porque suelen tener derechos de autor: `cajas_texto.json` guarda solo las cajas (con el nombre de la carpeta y de los archivos) y `carpetas_locales.json` (fuera del repositorio) dónde está cada serie en ese equipo. Quien no tiene las imágenes puede reentrenar igual con las medidas de `candidatos_propios.csv`, que son números; solo hacen falta si cambia la búsqueda de candidatos (`VERSION_RASGOS`).

Al cerrar, la ventana ofrece entrenar (`python -m evaluacion.entrenar_detector`). El informe muestra, por serie, los **candidatos** (qué parte del texto llega a proponerse: si es bajo, entrenar no lo arregla) y la precisión, el recall y el F1 sin haber visto esa serie.

## Archivos

| Archivo | Contenido |
|---|---|
| `sinteticos.py` | Generador de pares sintéticos. `python -m evaluacion.sinteticos` guarda algunos como PNG |
| `metricas.py` | IoU, Dice, MAE, PSNR, SSIM, F1 de detección, macro-F1 y bootstrap |
| `evaluar.py` | Corre todo y escribe `resultados/` |
| `calibrar_selector.py` | Calcula los umbrales del selector y su validación cruzada por obra (`python -m evaluacion.calibrar_selector`) |
| `etiquetar.py` | Ventana para etiquetar regiones reales |
| `muestreo.py` | Estratos, orden de etiquetado y pesos |
| `entrenar_detector.py` | Arma los candidatos, mide el detector con validación cruzada por obra y guarda el modelo |
| `horizontales.py` | Páginas de OpenMantra con los globos traducidos en renglones (datos de entrenamiento simulados) |
| `marcar.py`, `cajas.py` | Programa para marcar el texto de páginas propias, y el formato en que se guarda |
| `candidatos.csv.gz` | Candidatos de OpenMantra con sus medidas y si son texto, comprimido (se genera solo) |
| `cajas_texto.json` | Cajas de texto marcadas en páginas propias (sin imágenes) |
| `candidatos_propios.csv` | Candidatos de las páginas propias con sus medidas, para entrenar sin las imágenes |
| `estratos.csv` | Medida y estrato de cada caja de texto de OpenMantra (se genera solo) |
| `etiquetas.csv` | Etiquetas hechas a mano: región, clase y quién la etiquetó |
| `resultados/resumen.md` | Tablas listas para el informe |
| `resultados/*.csv` | Valores por par, por página y por región |
| `resultados/ejemplos.png` | Lámina con pares de ejemplo: entrada, referencia, máscara, resultado y fondo limpio |
| `fuentes/` | Noto Sans JP Bold y su licencia |

## Créditos

- **Noto Sans JP** (Google, [github.com/notofonts/noto-cjk](https://github.com/notofonts/noto-cjk)), licencia SIL Open Font License 1.1, incluida en [`fuentes/OFL.txt`](fuentes/OFL.txt).
- Las frases de los pares sintéticos y las páginas reales son del **OpenMantra dataset** (CC BY-NC 4.0); créditos y cita en [`../datasets/README.md`](../datasets/README.md).
