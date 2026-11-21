# Manual de instalación y usuario · PanelClean 0.3.1-beta

PanelClean es una aplicación de escritorio para borrar el texto de los globos de manga y reconstruir el fondo que había detrás. Tú marcas dónde están las letras; la aplicación separa los trazos de las letras del dibujo (la *máscara*), reconoce qué tipo de fondo hay detrás y elige cómo rellenarlo. Antes de guardar puedes revisar, corregir y deshacer todo.

Proyecto del curso TEL-328 Procesamiento Digital de Imágenes (UTFSM, 2026).

**Contenido**

1. [Requisitos](#1-requisitos)
2. [Instalación](#2-instalación)
3. [La ventana](#3-la-ventana)
4. [Limpiar una página paso a paso](#4-limpiar-una-página-paso-a-paso)
5. [Cuando la máscara falla](#5-cuando-la-máscara-falla)
6. [Detectar globos (experimental)](#6-detectar-globos-experimental)
7. [Ajustes avanzados](#7-ajustes-avanzados)
8. [Atajos de teclado y ratón](#8-atajos-de-teclado-y-ratón)
9. [Limitaciones conocidas](#9-limitaciones-conocidas)
10. [Solución de problemas](#10-solución-de-problemas)
11. [Créditos](#11-créditos)

---

## 1. Requisitos

| | |
|---|---|
| Sistema | Windows 10 u 11, o Linux con escritorio |
| Python | 3.10 o superior |
| Pantalla | 1366 × 768 o más |
| Espacio | Unos 50 MB la carpeta del proyecto y unos 200 MB más al instalar |
| Internet | Solo la primera vez, para descargar NumPy, OpenCV y Pillow |

Todo lo que instala PanelClean queda dentro de su propia carpeta (en `.venv`). No modifica el Python del sistema.

## 2. Instalación

### Windows

1. **Instala Python** desde [python.org](https://www.python.org/downloads/). En la primera pantalla del instalador marca **«Add Python to PATH»**. Si ya tienes Python 3.10 o superior, salta este paso.
2. **Descarga PanelClean** desde [github.com/Tecktomi/PanelClean](https://github.com/Tecktomi/PanelClean): botón **Code → Download ZIP**, y descomprime **todo** el ZIP en una carpeta (por ejemplo, en el Escritorio). También sirve `git clone https://github.com/Tecktomi/PanelClean.git`.
3. **Haz doble clic en `INICIAR_WINDOWS.bat`**. La primera vez se abre una consola que prepara el entorno e instala las bibliotecas; tarda unos minutos. Si la instalación se corta (por ejemplo, por la conexión), basta con volver a abrir el archivo: retoma sola.
4. Las veces siguientes, el mismo doble clic abre la aplicación directamente.

Si ves esta ventana con la página de ejemplo, PanelClean quedó instalado:

![PanelClean recién abierto, con la página de ejemplo](docs/manual/ventana_inicial.png)

### Linux

En Debian o Ubuntu, instala primero Python, su módulo de entornos virtuales y Tkinter:

```bash
sudo apt install python3 python3-venv python3-tk
```

Luego, en una terminal, dentro de la carpeta del proyecto:

```bash
chmod +x INICIAR_LINUX.sh
./INICIAR_LINUX.sh
```

El lanzador avisa si falta algo y dice cómo instalarlo. Nota: el lanzador de Linux todavía no se ha probado en un equipo con Linux; si falla, avísanos con el mensaje que muestre.

### Opciones

- **Instalar sin abrir la aplicación:** `INICIAR_WINDOWS.bat --instalar` (en Linux, `./INICIAR_LINUX.sh --instalar`).
- **Desinstalar:** borra la carpeta de PanelClean. No queda nada fuera de ella.

## 3. La ventana

![Partes de la ventana, numeradas](docs/manual/interfaz.png)

| # | Parte | Para qué sirve |
|---|---|---|
| 1 | Barra superior | **Ejemplo** abre la página incluida para practicar; **Abrir imagen** abre una página tuya; **Exportar** guarda el resultado |
| 2 | Regiones | Lista de los rectángulos marcados. ✓ = ya limpiada · • = tiene máscara. **Quitar** borra las regiones elegidas y **Crear máscaras** las analiza |
| 3 | Edición | Herramienta activa: **Seleccionar** (marcar regiones), **Añadir máscara** y **Borrar máscara** (pincel). **Ver máscara** muestra u oculta el rosa |
| 4 | Ajustes avanzados | Afinan cómo se crea la máscara y el tamaño del pincel (sección 7) |
| 5 | Vista ORIGINAL | La página sin cambios. Aquí se marcan las letras y se ve la máscara en rosa |
| 6 | Vista RESULTADO | La página con las limpiezas aplicadas |
| 7 | Navegación | Rueda del ratón: zoom en ambas vistas a la vez · clic derecho y arrastrar: mover · **Ajustar vista**: volver a ver la página completa |
| 8 | Selección | Datos de la región elegida: clase de fondo y las medidas con que se decidió |
| 9 | Limpieza | **Método**, **Radio de inpainting** y los botones **Limpiar selección**, **Limpiar todas** y **Deshacer** |
| 10 | Estado | Qué acaba de pasar y qué conviene hacer después. Se repite en la barra inferior |

Pasa el cursor sobre un botón para ver una ayuda breve.

## 4. Limpiar una página paso a paso

Para practicar, usa **Ejemplo**: la página tiene un globo blanco, uno sobre una viñeta punteada y uno con trama.

### Paso 1 · Abrir la página

Pulsa **Abrir imagen** (o `Ctrl + O`) y elige el archivo. Acepta PNG, JPG, BMP y WebP. Si la imagen mide más de 3500 px de lado, PanelClean pregunta antes de abrirla, porque procesarla toma más tiempo.

### Paso 2 · Marcar las letras

Con la herramienta **Seleccionar**, arrastra en la vista ORIGINAL un rectángulo **ajustado a las letras**, sin incluir el contorno del globo. Cada rectángulo es una región (R01, R02…). Puedes marcar varias.

![Tres regiones marcadas sobre la página de ejemplo](docs/manual/regiones.png)

- Un clic sobre una región la elige; `Ctrl + clic` la suma o la quita de la selección. En la lista, `Ctrl` o `Shift + clic` eligen varias.
- Las regiones elegidas se ven en celeste; las demás, en naranja.
- Un rectángulo más chico que 12 × 12 px no se acepta.
- Si un globo tiene texto en dos zonas separadas, marca dos regiones: cuanto menos dibujo quede dentro, mejor sale la máscara.

### Paso 3 · Crear las máscaras

Pulsa **Crear máscaras**. Las letras que PanelClean va a borrar se tiñen de rosa.

![Máscaras creadas: las letras se ven en rosa](docs/manual/mascaras.png)

**Revisa el rosa antes de limpiar.** Todo lo rosa se va a borrar y reconstruir. Si cubre parte del dibujo o deja letras sin cubrir, corrígelo (sección 5).

### Paso 4 · Revisar el tipo de fondo

El panel **Selección** muestra la clase de fondo que PanelClean encontró detrás de las letras. Con el método **Automático**, la clase decide cómo se limpia:

| Clase | Qué hay detrás de las letras | Qué hace el modo Automático |
|---|---|---|
| **Uniforme** | Papel liso, blanco o gris | **Relleno local**: pinta con el tono del papel que rodea a las letras |
| **Estructurado** | Trama de puntos, rayado o degradado, sin negro | **Inpainting Telea**: reconstruye el fondo a partir de sus bordes |
| **Complejo / revisar** | Dibujo, con trazos negros | No limpia: queda para que la revises (sección 5) |
| **Revisar máscara** (en naranja) | La máscara cubre más del 36 % de la región, o la región ocupa más del 5 % de la página (y más de 40 000 px) | No limpia: probablemente la máscara incluye dibujo |

Las medidas del panel explican la decisión: la **entropía** indica cuántos tonos distintos hay en el fondo (pocos = papel liso), la **textura** qué parte del fondo tiene trama, y el **tono oscuro** cuán oscuro es lo más oscuro del fondo (cerca de 0 = hay tinta de dibujo).

Si no estás de acuerdo, elige otro **Método** en el panel Limpieza: se aplica a las regiones elegidas.

### Paso 5 · Limpiar

- **Limpiar selección** limpia las regiones elegidas en la lista.
- **Limpiar todas** intenta limpiar todas; las que quedan para revisión no se tocan.

La barra de estado dice cuántas se limpiaron, con qué método y cuáles quedaron pendientes. La limpieza **nunca cambia un píxel fuera de la máscara**. Con la rueda del ratón puedes acercarte para comparar ambas vistas:

![De cerca: el globo blanco antes y después](docs/manual/detalle_globo_liso.png)

### Paso 6 · Comparar, deshacer y quitar

- **Deshacer** (`Ctrl + Z`) vuelve atrás la última limpieza o la última región quitada, hasta 15 pasos (menos en páginas muy grandes, para no llenar la memoria).
- **Quitar** (`Supr`) borra las regiones elegidas y también su limpieza del resultado.
- Puedes volver a limpiar una región con otro método: el resultado se recalcula siempre desde la página original, así que las limpiezas no se acumulan ni se degradan.

### Paso 7 · Exportar

Pulsa **Exportar** (`Ctrl + S`) y elige dónde guardar. Se crean tres archivos; tu imagen original no se modifica:

| Archivo | Contenido |
|---|---|
| `nombre.png` | La página limpia |
| `nombre_mascara.png` | La máscara de todas las regiones limpiadas (blanco = píxel reconstruido) |
| `nombre_reporte.json` | Por región: posición, si se limpió, clase de fondo, método usado, medidas del fondo y píxeles de la máscara |

## 5. Cuando la máscara falla

La máscara se calcula por el tono, el tamaño y la forma de los trazos, no por el significado de las letras. PanelClean ya descarta por sí solo lo que entra por el borde de la selección y sigue fuera de ella (una sombra, el borde del globo, un trazo de dibujo), pero todavía puede equivocarse de dos maneras.

**Faltan letras o signos.** Las piezas muy pequeñas, como puntos o comas, pueden quedar sin rosa. En este ejemplo real, la máscara cubre «かっ» pero no los puntos suspensivos:

![La máscara de «かっ…» no cubre los puntos suspensivos](docs/manual/mascara_incompleta.png)

Para corregirla:

1. Elige la región y la herramienta **Añadir máscara**. El tamaño del pincel está en Ajustes avanzados.
2. Pinta sobre lo que falta. Al soltar el botón, el fondo se vuelve a analizar.
3. Limpia otra vez. En este caso, el rectángulo toca el fondo negro de fuera del globo, así que el análisis dice **Complejo / revisar** y el modo automático no lo limpia. Como detrás de las letras hay papel blanco, elige **Relleno local** en Método y pulsa **Limpiar selección**.

![Puntos añadidos con el pincel y región limpiada con Relleno local](docs/manual/mascara_completada.png)

Si después de limpiar quedan motas sueltas, como aquí junto a la «っ», un toque más con **Añadir máscara** y otra limpieza las borran.

**Sobra dibujo.** Si dentro de la selección hay un trazo de dibujo tan oscuro y delgado como una letra, puede quedar marcado. Bórralo con la herramienta **Borrar máscara** antes de limpiar. Esto importa más de lo que parece: si el dibujo queda dentro de la máscara, lo que rodea a las letras parece papel liso, la clase dice **Uniforme** y la limpieza pintaría ese dibujo de blanco. Por eso conviene revisar siempre el rosa.

**Otras formas de arreglarlo:**

- **Ajustar el rectángulo**: quita la región y márcala de nuevo más ajustada, o divídela en dos.
- **Sensibilidad** (Ajustes avanzados): súbela si faltan signos pequeños en muchas regiones.
- **Crear máscaras** otra vez descarta las correcciones hechas con el pincel; por eso pregunta antes.
- **Dejarla para el final**: una región en «Complejo / revisar» no se limpia sola. Puedes elegir un método a mano o limpiarla en otro editor.

## 6. Detectar globos (experimental)

**Detectar globos** busca contornos ovalados con tinta adentro y los agrega como regiones. Es un atajo para páginas con muchos globos, pero **todavía se equivoca a menudo**: en las pruebas con páginas reales encontró alrededor de la mitad de los textos, y cerca de la mitad de lo que propuso no era un globo con texto. Por ejemplo, puede proponer viñetas completas, como R01 y R03 aquí:

![Candidatos de Detectar globos: R01 y R03 son viñetas, no globos](docs/manual/detectar_globos.png)

Una región así es demasiado grande para ser solo texto, y su máscara incluye dibujo. Por eso, las regiones que ocupan más del 5 % de la página quedan en **Revisar máscara** y el modo automático no las limpia. En las pruebas con páginas reales, esto evita el 90 % del dibujo que se habría borrado al limpiar todas las detecciones sin revisarlas. Aun así, el resto todavía puede incluir algo de dibujo. **Antes de limpiar:**

1. Elige los candidatos que no son globos y pulsa **Quitar**.
2. Marca a mano el texto que el detector no encontró.
3. Crea las máscaras, revisa el rosa y limpia.

Con los candidatos revisados y el resto del texto marcado a mano, la página queda así:

![Página real limpiada tras revisar los candidatos](docs/manual/pagina_real_limpia.png)

## 7. Ajustes avanzados

Se abren con **Ajustes avanzados** en el panel izquierdo. Después de cambiar uno, pulsa **Crear máscaras** para recalcular.

![Ajustes avanzados abiertos](docs/manual/ajustes_avanzados.png)

| Ajuste | Qué hace | Cuándo cambiarlo |
|---|---|---|
| **Umbralización** | Cómo se separa la tinta del fondo. **Otsu** usa un único umbral para toda la región; **Adaptativo** calcula uno distinto en cada zona | Adaptativo, si el fondo es un degradado o está más oscuro de un lado |
| **Sensibilidad** (1-8, por defecto 4) | Tamaño mínimo de una pieza de tinta para considerarla parte de una letra: 7 píxeles de área con 1, 3 con 8 | Súbela si faltan puntos, comas o marcas pequeñas; bájala si aparecen motas sueltas |
| **Expansión de máscara** (0-3, por defecto 2) | Cuántos píxeles crece la máscara alrededor de las letras, solo sobre tinta, para no dejar un borde gris | Súbela si queda el contorno de las letras tras limpiar; bájala si la limpieza se come detalles del fondo |
| **Tamaño del pincel** (2-20) | Radio del pincel de Añadir y Borrar máscara, en píxeles de la imagen | Chico para detalles, grande para zonas amplias |
| **Radio de inpainting** (1-8, panel Limpieza) | Cuánto alrededor mira el inpainting para reconstruir | Más alto suaviza más; más bajo conserva mejor los bordes |

## 8. Atajos de teclado y ratón

| Acción | Atajo |
|---|---|
| Abrir imagen | `Ctrl + O` |
| Exportar | `Ctrl + S` |
| Deshacer | `Ctrl + Z` |
| Quitar regiones elegidas | `Supr` |
| Volver a la herramienta Seleccionar | `Esc` |
| Zoom | Rueda del ratón sobre una vista |
| Mover la vista | Clic derecho y arrastrar |
| Elegir varias regiones | `Ctrl + clic` en la vista; `Ctrl` o `Shift + clic` en la lista |

## 9. Limitaciones conocidas

PanelClean es una beta. Estas son sus limitaciones medidas; el detalle está en [`evaluacion/resultados/resumen.md`](evaluacion/resultados/resumen.md).

- **Tramas y degradados.** El inpainting no reproduce bien un patrón regular: donde estaban las letras queda una zona más clara o borrosa.

  ![De cerca: el globo con trama antes y después](docs/manual/detalle_globo_trama.png)

- **Texto sobre dibujo.** La máscara confunde con letras los trazos de dibujo de grosor parecido (sección 5), y cuando las letras tienen un borde blanco suele quedar su silueta. Es el caso con peores resultados en las pruebas.
- **Letras blancas sobre fondo negro.** No se detectan: la máscara busca tinta oscura. (El texto negro con borde blanco sobre un fondo oscuro sí se detecta.)
- **Detector de globos.** Es experimental (sección 6).
- **Clasificación del fondo.** Acierta bien en papel liso, pero con tramas finas reales se equivoca más que con dibujo. Revisa la clase en el panel Selección.
- **Restos.** En algunos globos queda un moteado gris muy tenue donde estaban las letras, sobre todo en páginas con mucha compresión JPEG. Se nota al mirar de cerca y se borra con **Añadir máscara** o subiendo la **Expansión**. También pueden quedar signos muy pequeños sin borrar (sección 5).

## 10. Solución de problemas

| Problema | Qué hacer |
|---|---|
| «No se encontró Python 3.10 o superior» | Instala Python desde python.org marcando «Add Python to PATH» y vuelve a abrir `INICIAR_WINDOWS.bat` |
| «No se pudieron instalar las dependencias» | Revisa la conexión a Internet y vuelve a abrir el lanzador: la instalación se reintenta sola |
| Linux: «Falta Tkinter» | `sudo apt install python3-tk python3-venv` |
| `ModuleNotFoundError: No module named 'PIL'` (o `cv2`) al correr un `.py` | Se usó el Python del sistema. Abre PanelClean con el lanzador, o usa `.venv\Scripts\python` (Linux: `.venv/bin/python`) |
| Una región no se limpia | Está en «Complejo / revisar» o «Revisar máscara», o su máscara quedó vacía. Revisa el rosa, corrígelo o elige un método a mano |
| Al arrastrar no se crea un rectángulo | La herramienta activa es un pincel. Pulsa `Esc` o elige **Seleccionar** |
| El pincel no hace nada | Hay que elegir antes una región (un clic sobre ella) y pintar dentro de su rectángulo |
| La ventana no cabe en la pantalla | En pantallas chicas PanelClean se abre maximizada; los paneles laterales se pueden angostar arrastrando su borde |
| Va lento | Pasa con páginas muy grandes. Una página de 2048 px se procesa en menos de 2 s en un portátil común |

## 11. Créditos

- **Equipo:** Jesús Ignacio Bustamante Llanquitrue, Patricio Marcelo Guzmán Aranda, Sofia Victoria Nicolao Madrid y Tomás Ramdohr Arellano.
- **Páginas reales de este manual:** obras «tencho_isoro» y «tojime_no_siora» del **OpenMantra dataset** (R. Hinami, S. Ishiwatari, K. Yasuda y Y. Matsui, AAAI 2021), licencia CC BY-NC 4.0. Créditos completos en [`datasets/README.md`](datasets/README.md).
- **Página de ejemplo** (`examples/demo_pagina.png`): sintética, hecha por el equipo.
- Bibliotecas: OpenCV, NumPy, Pillow y Tkinter.
