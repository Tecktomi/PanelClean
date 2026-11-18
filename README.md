# PanelClean - Hito 2

PanelClean es una beta de escritorio para apoyar la limpieza de texto dentro de globos de manga. La aplicación permite seleccionar las letras, crear una máscara, aplicar una limpieza y comparar el resultado antes de exportarlo.

## Ejecutar en Windows

1. Descomprime todo el ZIP en una carpeta.
2. Debes tener Python 3.10 o superior instalado.
3. Haz doble clic en `INICIAR_WINDOWS.bat`.
4. La primera ejecución instala automáticamente las dependencias y puede tardar un poco más.

## Ejecutar en Linux

En una terminal, dentro de la carpeta del proyecto:

```bash
chmod +x INICIAR_LINUX.sh
./INICIAR_LINUX.sh
```

En Debian/Ubuntu se necesitan `python3`, `python3-venv` y `python3-tk`.

## Uso básico

1. `Abrir imagen` para cargar una página o `Ejemplo` para usar la imagen incluida.
2. En la vista Original, arrastra un rectángulo ajustado alrededor de las letras que quieres limpiar.
3. Puedes seleccionar varias regiones.
4. Pulsa `Crear máscaras` y revisa la zona marcada.
5. Pulsa `Limpiar selección`.
6. Compara el resultado y usa `Deshacer` si es necesario.
7. Pulsa `Exportar` para guardar la página limpia.

## Estado de la beta

La versión actual demuestra el flujo principal del proyecto. Todavía puede fallar cuando la selección incluye líneas del dibujo, marcos de viñetas o fondos muy complejos. Por eso la máscara se muestra antes de aplicar la limpieza y existe corrección manual.

El informe breve `Informe_Hito2_PanelClean.pdf` resume el funcionamiento, la relación con TEL-328 y las mejoras pendientes.
