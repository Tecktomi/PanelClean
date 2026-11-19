#!/usr/bin/env bash
# Uso: ./INICIAR_LINUX.sh              instala lo necesario y abre PanelClean
#      ./INICIAR_LINUX.sh --instalar   solo instala (no abre la aplicación)
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[PanelClean] Falta Python 3. Instálalo con el gestor de paquetes de tu distribución."
  exit 1
fi

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "[PanelClean] Se necesita Python 3.10 o superior (tienes $(python3 --version))."
  exit 1
fi

if ! python3 -c "import tkinter" >/dev/null 2>&1; then
  echo "[PanelClean] Falta Tkinter."
  echo "Debian/Ubuntu: sudo apt install python3-tk python3-venv"
  echo "Arch: sudo pacman -S tk python"
  exit 1
fi

# Crear el entorno virtual la primera vez. Si falla a medias, se borra para
# que el siguiente intento empiece de cero.
if [ ! -x ".venv/bin/python" ]; then
  echo "[PanelClean] Preparando el entorno por primera vez..."
  if ! python3 -m venv .venv; then
    rm -rf .venv
    echo "No se pudo crear el entorno virtual. En Debian/Ubuntu instala: sudo apt install python3-venv"
    exit 1
  fi
fi

# Instalar dependencias si la instalación anterior no terminó bien.
# El archivo .venv/instalado.txt se crea solo cuando pip termina sin errores.
if [ ! -f ".venv/instalado.txt" ]; then
  echo "[PanelClean] Instalando dependencias, puede tardar unos minutos..."
  if ! .venv/bin/python -m pip install -r requirements.txt; then
    echo "[PanelClean] No se pudieron instalar las dependencias. Revisa tu conexión y vuelve a intentarlo."
    exit 1
  fi
  echo ok > .venv/instalado.txt
fi

if [ "${1:-}" = "--instalar" ]; then
  echo "[PanelClean] Instalación lista."
  exit 0
fi

exec .venv/bin/python main.py
