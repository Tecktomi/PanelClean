#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[PanelClean] Falta Python 3. Instálalo con el gestor de paquetes de tu distribución."
  exit 1
fi

if ! python3 -c "import tkinter" >/dev/null 2>&1; then
  echo "[PanelClean] Falta Tkinter."
  echo "Debian/Ubuntu: sudo apt install python3-tk python3-venv"
  echo "Arch: sudo pacman -S tk python"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[PanelClean] Preparando el entorno por primera vez..."
  if ! python3 -m venv .venv; then
    echo "No se pudo crear el entorno virtual. En Debian/Ubuntu instala: sudo apt install python3-venv"
    exit 1
  fi
  source .venv/bin/activate
  python -m pip install -r requirements.txt
else
  source .venv/bin/activate
fi

python main.py
