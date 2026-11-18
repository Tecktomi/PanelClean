@echo off
setlocal
cd /d "%~dp0"
title PanelClean - Hito 2

where py >nul 2>nul
if not errorlevel 1 (
    set "PY=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo.
        echo [PanelClean] No se encontro Python.
        echo Instala Python 3.10 o superior desde python.org y vuelve a ejecutar este archivo.
        echo Durante la instalacion marca la opcion "Add Python to PATH".
        echo.
        pause
        exit /b 1
    )
    set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [PanelClean] Preparando el entorno por primera vez...
    %PY% -m venv .venv
    if errorlevel 1 goto :venv_error
    call ".venv\Scripts\activate.bat"
    python -m pip install -r requirements.txt
    if errorlevel 1 goto :deps_error
) else (
    call ".venv\Scripts\activate.bat"
)

python main.py
if errorlevel 1 (
    echo.
    echo [PanelClean] La aplicacion se cerro con un error.
    pause
)
exit /b 0

:venv_error
echo.
echo [PanelClean] No se pudo crear el entorno virtual de Python.
pause
exit /b 1

:deps_error
echo.
echo [PanelClean] No se pudieron instalar las dependencias.
echo Revisa tu conexion a Internet e intenta nuevamente.
pause
exit /b 1
