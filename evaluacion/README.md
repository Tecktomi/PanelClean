# Evaluación de PanelClean

Mide, de forma reproducible, las metas comprometidas en el Hito 1 (presentación, diapositiva 7 y respaldo B). Los resultados más recientes están en [`resultados/resumen.md`](resultados/resumen.md).

## Cómo correrla

Desde la raíz del repositorio, con el entorno ya instalado:

```bash
.venv\Scripts\python -m evaluacion.evaluar      # Windows
.venv/bin/python -m evaluacion.evaluar          # Linux
```

Tarda alrededor de un minuto en un portátil. Con `--rapido` corre una versión corta para probar. No necesita instalar nada más que lo de `requirements.txt`.

## Qué se mide

| # | Métrica | Tarea | Datos | Definición | Meta (Hito 1) |
|---|---|---|---|---|---|
| 1 | Detección: precisión, recall, F1 | T1.2.3 | OpenMantra, 214 páginas | Una caja de texto está acertada si ≥ 80 % de su área queda dentro de una región detectada; una detección es correcta si contiene así algún texto. Se informa también el F1 estándar (uno a uno, IoU ≥ 0,5), y cuánto dibujo borraría el modo automático si se limpiaran todas las detecciones sin revisarlas | F1 ≥ 0,85 |
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

**OpenMantra** (`datasets/open-mantra`). Detección, rendimiento y selector con regiones reales. La obra `tencho_isoro` está reservada como conjunto de prueba: si más adelante se ajusta o entrena algo, no se usa para eso.

## Decisiones de diseño

- **Máscara del sistema y máscara perfecta.** La limpieza se mide dos veces. Con la máscara perfecta se aísla la elección del método; con la del sistema se ve lo que obtiene el usuario.
- **Tres métricas para la hipótesis.** El MAE favorece por construcción el relleno con la mediana (la mediana minimiza el error absoluto) aunque borre la textura; el SSIM sí mide si la textura se conserva.
- **Adaptativo comparable en todos los pares.** La aplicación deja los fondos complejos para revisión manual. Para comparar en los mismos pares, el adaptativo usa Relleno local si el fondo se clasifica como Uniforme y Telea en otro caso. Aparte se informa cuántos pares deja la aplicación para revisión.
- **Piso y techo.** *Sin limpiar* es el piso; el *oráculo por par* (el mejor método único en cada par) es el techo de cualquier selector.
- **Selector calibrado sin mirar la prueba.** La regla mide el fondo en una franja de hasta 14 px alrededor de las letras: la *entropía* del histograma separa el papel liso del resto, y el *tono oscuro* (percentil 10) separa el dibujo, que tiene tinta casi negra, de la trama o el degradado, que no la tienen. Una tercera medida, la *textura repartida* (fracción de la franja que varía en ventanas de 7 × 7 px), evita confundir con papel una trama rala que cubre toda la franja. Los umbrales se calculan con `calibrar_selector.py` a partir de pares sintéticos de desarrollo (semilla 7) y de las regiones reales etiquetadas, sin la obra de prueba. Como las regiones reales sirven para ajustar y para medir, el valor que cuenta es la validación cruzada por obra: se ajusta sin una obra y se mide en ella. La varianza y la densidad de bordes de la regla anterior se descartaron porque en manga real las tramas son finas y de poco contraste: la regla las tomaba por papel liso y, en los sintéticos, tomaba las tramas por dibujo.
- **Zona aproximada sin mirar la prueba.** `texto_en_zona` toma como letras lo oscuro que el papel blanco de un globo rodea por completo, así el fondo negro o la trama de alrededor no desvían el umbral. Sus parámetros se ajustaron con zonas simuladas de las cuatro obras de desarrollo. La obra de prueba se midió una vez con la primera versión; después se agregó, por la página de ejemplo, el descarte de tramas y de contornos de globo, y se volvió a medir. Como las cajas de OpenMantra son holgadas, no sirve medir el IoU contra ellas: la referencia son las letras que da la máscara de la caja anotada.
- **Detección por cobertura.** Para limpiar, lo que importa es que el texto quede dentro de la región; el IoU castiga que el globo detectado sea más grande que el texto. Por eso el criterio principal es la cobertura, y el IoU ≥ 0,5 se informa como referencia.
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
- Todas las personas ven las regiones en el mismo orden. Si dos personas etiquetan las mismas, la evaluación informa cuánto coinciden y deja fuera las regiones en desacuerdo hasta resolverlas.
- Para revisar las regiones que uno ya marcó como no lisas: `ETIQUETAR_WINDOWS.bat TuNombre --repasar` (o `--repasar` en el comando de arriba).
- Después de etiquetar se vuelve a correr `python -m evaluacion.evaluar`.

**Por qué no salen al azar.** En OpenMantra casi todo el texto está en globos blancos: en 107 regiones mostradas al azar, 103 eran lisas. Para no tener que revisar el dataset entero, `muestreo.py` reparte las 1592 regiones en 4 estratos según cuántos grises intermedios (ni tinta ni papel) hay dentro de la caja, y muestra más seguido las del estrato más cargado. Dentro de cada estrato el orden es al azar. La evaluación pondera cada región por N_h / n_h (regiones del estrato en todo el dataset ÷ regiones etiquetadas de ese estrato), así el resultado estima el desempeño en todo OpenMantra y no solo en las regiones mostradas.

## Archivos

| Archivo | Contenido |
|---|---|
| `sinteticos.py` | Generador de pares sintéticos. `python -m evaluacion.sinteticos` guarda algunos como PNG |
| `metricas.py` | IoU, Dice, MAE, PSNR, SSIM, F1 de detección, macro-F1 y bootstrap |
| `evaluar.py` | Corre todo y escribe `resultados/` |
| `calibrar_selector.py` | Calcula los umbrales del selector y su validación cruzada por obra (`python -m evaluacion.calibrar_selector`) |
| `etiquetar.py` | Ventana para etiquetar regiones reales |
| `muestreo.py` | Estratos, orden de etiquetado y pesos |
| `estratos.csv` | Medida y estrato de cada caja de texto de OpenMantra (se genera solo) |
| `etiquetas.csv` | Etiquetas hechas a mano: región, clase y quién la etiquetó |
| `resultados/resumen.md` | Tablas listas para el informe |
| `resultados/*.csv` | Valores por par, por página y por región |
| `resultados/ejemplos.png` | Lámina con pares de ejemplo: entrada, referencia, máscara, resultado y fondo limpio |
| `fuentes/` | Noto Sans JP Bold y su licencia |

## Créditos

- **Noto Sans JP** (Google, [github.com/notofonts/noto-cjk](https://github.com/notofonts/noto-cjk)), licencia SIL Open Font License 1.1, incluida en [`fuentes/OFL.txt`](fuentes/OFL.txt).
- Las frases de los pares sintéticos y las páginas reales son del **OpenMantra dataset** (CC BY-NC 4.0); créditos y cita en [`../datasets/README.md`](../datasets/README.md).
