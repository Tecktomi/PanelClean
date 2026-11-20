@echo off
setlocal
cd /d "%~dp0"
title PanelClean - Etiquetar fondos

rem Abre la ventana para etiquetar el fondo de regiones reales de OpenMantra.
rem Uso: doble clic (pide el nombre), o  ETIQUETAR_WINDOWS.bat TuNombre
rem      ETIQUETAR_WINDOWS.bat TuNombre --repasar   revisa tus regiones no lisas
rem Las etiquetas se guardan en evaluacion\etiquetas.csv.

rem 1) Preparar el entorno si todavia no existe (lo mismo que INICIAR_WINDOWS.bat).
if not exist ".venv\instalado.txt" (
    call INICIAR_WINDOWS.bat --instalar
    if errorlevel 1 exit /b 1
)

rem 2) Pedir el nombre de quien etiqueta.
set "NOMBRE=%~1"
if not defined NOMBRE set /p "NOMBRE=Tu nombre: "
if not defined NOMBRE goto :sin_nombre

rem 3) Abrir la ventana con el Python del proyecto.
".venv\Scripts\python.exe" -m evaluacion.etiquetar --nombre "%NOMBRE%" %2
if errorlevel 1 goto :error
exit /b 0

:sin_nombre
echo.
echo [PanelClean] Hace falta un nombre para guardar tus etiquetas.
pause
exit /b 1

:error
echo.
echo [PanelClean] No se pudo abrir la ventana de etiquetado. Revisa el mensaje de arriba.
pause
exit /b 1
