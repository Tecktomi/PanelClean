@echo off
setlocal
cd /d "%~dp0"
title PanelClean

rem Uso: INICIAR_WINDOWS.bat               instala lo necesario y abre PanelClean
rem      INICIAR_WINDOWS.bat --instalar    solo instala (no abre la aplicacion)

rem 1) Buscar un Python 3.10 o superior que realmente funcione.
rem    Se prueba ejecutandolo, porque Windows trae un acceso directo "python"
rem    de Microsoft Store que existe aunque Python no este instalado.
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto :no_python

rem 2) Crear el entorno virtual la primera vez.
if not exist ".venv\Scripts\python.exe" (
    echo [PanelClean] Preparando el entorno por primera vez...
    %PY% -m venv .venv
    if errorlevel 1 goto :venv_error
)

rem 3) Instalar dependencias si la instalacion anterior no termino bien.
rem    El archivo .venv\instalado.txt se crea solo cuando pip termina sin errores.
if not exist ".venv\instalado.txt" (
    echo [PanelClean] Instalando dependencias, puede tardar unos minutos...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :deps_error
    echo ok> ".venv\instalado.txt"
)

if /i "%~1"=="--instalar" (
    echo [PanelClean] Instalacion lista.
    exit /b 0
)

".venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo [PanelClean] La aplicacion se cerro con un error.
    pause
)
exit /b 0

:no_python
echo.
echo [PanelClean] No se encontro Python 3.10 o superior.
echo Instala Python desde python.org y vuelve a ejecutar este archivo.
echo Durante la instalacion marca la opcion "Add Python to PATH".
echo.
pause
exit /b 1

:venv_error
echo.
echo [PanelClean] No se pudo crear el entorno virtual de Python.
pause
exit /b 1

:deps_error
echo.
echo [PanelClean] No se pudieron instalar las dependencias.
echo Revisa tu conexion a Internet y vuelve a ejecutar este archivo:
echo la instalacion se reintentara automaticamente.
pause
exit /b 1
