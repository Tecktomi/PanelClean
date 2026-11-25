"""Evaluación de PanelClean: pares sintéticos, métricas y script de resultados."""

import importlib.util
import sys

# Si se ejecuta con un Python que no es el del proyecto (por ejemplo, «python»
# en vez de «.venv\Scripts\python»), faltan las bibliotecas. Mejor explicarlo
# que mostrar un error de importación.
_FALTAN = [m for m in ('cv2', 'numpy', 'PIL') if importlib.util.find_spec(m) is None]
if _FALTAN:
    sys.exit(
        f'Faltan bibliotecas ({", ".join(_FALTAN)}): este no es el Python del proyecto.\n'
        'Ejecuta los comandos con el Python del entorno, desde la carpeta PanelClean:\n'
        '  Windows: .venv\\Scripts\\python -m evaluacion.etiquetar --nombre TuNombre\n'
        '           .venv\\Scripts\\python -m evaluacion.evaluar\n'
        '  Linux:   .venv/bin/python -m evaluacion.etiquetar --nombre TuNombre\n'
        'Para etiquetar en Windows basta con hacer doble clic en ETIQUETAR_WINDOWS.bat.\n'
        'Si la carpeta .venv no existe, ejecuta antes INICIAR_WINDOWS.bat '
        '(o bash INICIAR_LINUX.sh).'
    )
