@echo off
setlocal
cd /d "%~dp0"
title PanelClean - Marcar texto

rem Abre el programa para marcar donde hay texto en paginas propias. Con eso se le
rem ensena al detector otros estilos e idiomas, y se mide como le va en ellos.
rem Uso: doble clic (pide las paginas y el nombre), o arrastra encima de este archivo
rem      un PDF o la carpeta con las paginas de UNA serie (imagenes o PDF).
rem Las cajas se guardan en evaluacion\cajas_texto.json; las imagenes no se copian.

rem 1) Preparar el entorno si todavia no existe (lo mismo que INICIAR_WINDOWS.bat).
fc requirements.txt ".venv\instalado.txt" >nul 2>nul
if errorlevel 1 (
    call INICIAR_WINDOWS.bat --instalar
    if errorlevel 1 exit /b 1
)

rem 2) Abrir la ventana con el Python del proyecto.
if "%~1"=="" (
    ".venv\Scripts\python.exe" -m evaluacion.marcar
) else (
    ".venv\Scripts\python.exe" -m evaluacion.marcar "%~1"
)
if errorlevel 1 goto :error
echo.
pause
exit /b 0

:error
echo.
echo [PanelClean] El programa termino con un error. Revisa el mensaje de arriba.
pause
exit /b 1
