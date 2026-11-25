# PanelClean

Aplicación de escritorio para limpiar el texto de los globos de manga: separa las letras del dibujo, reconoce el tipo de fondo que hay detrás y elige cómo reconstruirlo. Antes de guardar se puede revisar y corregir todo, y deshacer las limpiezas.

Proyecto del curso TEL-328 Procesamiento Digital de Imágenes (UTFSM, 2026) · versión 0.3.1-beta.

- **[Manual de instalación y usuario](MANUAL.md)**, con capturas (también en PDF: [`docs/Manual_PanelClean.pdf`](docs/Manual_PanelClean.pdf)).
- **[Documento de análisis y diseño](docs/Documento_Analisis_Diseno_PanelClean.pdf)** (PDF): requisitos, planificación, arquitectura, módulos, interfaces y resultados.
- **[Resultados de la evaluación](evaluacion/resultados/resumen.md)** frente a las metas del Hito 1.
- **Bitácora** del proyecto: [`Bitacora_PanelClean.xlsx`](Bitacora_PanelClean.xlsx).

## Ejecutar en Windows

1. Descarga el repositorio (`git clone` o «Code → Download ZIP» en GitHub y descomprime todo el ZIP).
2. Debes tener Python 3.10 o superior instalado. Al instalarlo desde python.org, marca «Add Python to PATH».
3. Haz doble clic en `INICIAR_WINDOWS.bat`.
4. La primera ejecución instala automáticamente las dependencias: tarda entre 15 segundos y un par de minutos, según el equipo y la conexión. Si la instalación se corta, al volver a abrir el archivo se reintenta sola.

Para instalar sin abrir la aplicación: `INICIAR_WINDOWS.bat --instalar` (en Linux, `bash INICIAR_LINUX.sh --instalar`).

## Ejecutar en Linux

Descomprime todo el ZIP y abre una terminal dentro de la carpeta del proyecto:

```bash
bash INICIAR_LINUX.sh
```

El doble clic no funciona en todos los escritorios; este comando, sí.

En Debian/Ubuntu se necesitan `python3`, `python3-venv` y `python3-tk`.

**Sin probar.** PanelClean se desarrolló y se probó solo en Windows 10 y 11. El lanzador de Linux está incluido y las bibliotecas que usa existen para Linux, pero el equipo no tuvo un computador donde ejecutarlo.

## Uso básico

El [manual](MANUAL.md) explica cada control, cómo corregir la máscara y qué hacer si algo falla. En resumen:

1. `Agregar hojas` (o `Abrir hojas`) para cargar una o varias páginas (imágenes o PDF), o `Ejemplo` para practicar con la imagen incluida.
2. Pulsa `Detectar texto` para encontrar el texto de la página: lo dudoso queda en naranja y no se limpia solo. Para agregar un globo que falte, haz clic sobre él o enciérralo a grandes rasgos (con `Ajustar al texto` la región se ajusta sola a las letras).
3. Pulsa `Crear máscaras` y revisa la zona marcada en rosa.
4. Pulsa `Limpiar selección`.
5. Compara el resultado y usa `Deshacer` si es necesario.
6. Pulsa `Exportar` para guardar la página limpia.

Con varias hojas abiertas aparece una tira de miniaturas: `Procesar todas` detecta y limpia el texto de cada hoja, una tras otra, y `Exportar todas` las guarda en una carpeta.

## Estado de la beta

La versión actual cubre el flujo completo. Todavía puede fallar con texto sobre dibujo o fondos muy complejos, y no detecta letras blancas sobre fondo negro. Por eso la máscara se muestra antes de aplicar la limpieza, existe corrección manual y el modo automático deja para revisión los casos dudosos. Las limitaciones medidas están en la sección 10 del [manual](MANUAL.md) y en los [resultados de la evaluación](evaluacion/resultados/resumen.md).

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `INICIAR_WINDOWS.bat`, `INICIAR_LINUX.sh` | Lanzadores: instalan lo necesario la primera vez y abren la aplicación |
| `main.py`, `requirements.txt` | Punto de entrada y dependencias (NumPy, OpenCV, Pillow y pypdfium2, para abrir PDF) |
| `panelclean/` | La aplicación: `vision.py` (detección de texto, máscara, clasificación del fondo y limpieza), `app.py` (interfaz), `pdf.py` (lectura de PDF) y `modelo_texto.yml` (el detector entrenado) |
| `tests/` | Pruebas automáticas |
| `evaluacion/` | Evaluación de las metas del Hito 1, con sus resultados en `evaluacion/resultados/` ([detalle](evaluacion/README.md)) |
| `ETIQUETAR_WINDOWS.bat` | Abre la herramienta para etiquetar el fondo de regiones reales (la usa la evaluación) |
| `MARCAR_WINDOWS.bat` | Abre el programa para marcar dónde hay texto en páginas propias: le enseña al detector otros estilos e idiomas y mide cómo le va en ellos |
| `examples/` | Página de ejemplo que abre el botón `Ejemplo` |
| `datasets/open-mantra/` | Páginas de manga reales usadas en la evaluación ([créditos](datasets/README.md)) |
| `docs/` | Entregas del curso (presentación del Hito 1, informe del Hito 2, documento final de análisis y diseño), el manual en PDF y sus capturas |
| `MANUAL.md`, `Bitacora_PanelClean.xlsx` | Manual de instalación y usuario, y bitácora del proyecto |

## Pruebas automáticas

Las pruebas están en `tests/` y usan solo la biblioteca estándar de Python. Con el entorno ya instalado, desde la carpeta del proyecto:

```bash
.venv\Scripts\python -m unittest discover tests      # Windows
.venv/bin/python -m unittest discover tests          # Linux
```

Comprueban, entre otras cosas, que la limpieza no modifique ningún píxel fuera de la máscara, que cada tipo de fondo se clasifique bien, que una viñeta entera no se limpie sola, que una zona holgada se ajuste al texto, que cada hoja de la cola conserve sus regiones y que «Quitar» y «Deshacer» funcionen juntos.

## Evaluación

La carpeta `evaluacion/` entrena el detector de texto (`python -m evaluacion.entrenar_detector`) y mide las metas del Hito 1: detección, máscara, selector, limpieza, conservación, tiempo por página y la hipótesis de que limpiar de forma adaptativa supera a un método único. Usa pares sintéticos (fondo limpio + texto japonés) y las páginas de OpenMantra.

```bash
.venv\Scripts\python -m evaluacion.evaluar      # Windows
.venv/bin/python -m evaluacion.evaluar          # Linux
```

Los resultados quedan en `evaluacion/resultados/resumen.md`. Para etiquetar el fondo de regiones reales (lo necesita la métrica del selector), en Windows basta con hacer doble clic en `ETIQUETAR_WINDOWS.bat`. El detalle de cada métrica está en [`evaluacion/README.md`](evaluacion/README.md).

## Documentos del curso

- [`docs/PanelClean_Presentacion_Hito1.pdf`](docs/PanelClean_Presentacion_Hito1.pdf): propuesta, arquitectura y plan de validación.
- [`docs/Informe_Hito2_PanelClean.pdf`](docs/Informe_Hito2_PanelClean.pdf): funcionamiento de la beta, relación con TEL-328 y mejoras pendientes.
- [`docs/Documento_Analisis_Diseno_PanelClean.pdf`](docs/Documento_Analisis_Diseno_PanelClean.pdf): documento de análisis y diseño de la entrega final, con los resultados y las limitaciones.
- [`docs/Manual_PanelClean.pdf`](docs/Manual_PanelClean.pdf): el manual de instalación y usuario en PDF.

## Datos incluidos y créditos

- `examples/demo_pagina.png`: página sintética de ejemplo que abre el botón `Ejemplo`.
- `datasets/open-mantra/`: **OpenMantra dataset**, de R. Hinami, S. Ishiwatari, K. Yasuda y Y. Matsui («Towards Fully Automated Manga Translation», AAAI 2021), descargado de <https://github.com/mantra-inc/open-mantra-dataset>. Son 214 páginas de cinco mangas japoneses con 1592 bloques de texto anotados, bajo licencia **CC BY-NC 4.0** (uso no comercial). No es obra del equipo; se incluye sin modificar para entrenar y evaluar la detección de texto. Los detalles, la licencia y la cita están en [`datasets/README.md`](datasets/README.md).
- `evaluacion/fuentes/NotoSansJP-Bold.otf`: fuente **Noto Sans JP** de Google ([github.com/notofonts/noto-cjk](https://github.com/notofonts/noto-cjk)), bajo licencia **SIL Open Font License 1.1** (incluida en `evaluacion/fuentes/OFL.txt`). Se usa para escribir el texto japonés de los pares sintéticos.
- Las capturas de páginas reales del manual también son de OpenMantra.
- `panelclean/fuentes/`: fuente **Quicksand** de la interfaz, de The Quicksand Project Authors ([github.com/andrew-paglinawan/QuicksandFamily](https://github.com/andrew-paglinawan/QuicksandFamily)), bajo licencia **SIL Open Font License 1.1** (incluida en `panelclean/fuentes/OFL.txt`). La aplicación la carga al iniciar, sin instalarla en el sistema.
