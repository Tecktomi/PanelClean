"""Ventana para etiquetar el fondo de regiones reales de OpenMantra (tarea T2.4.4).

Muestra, una por una, cajas de texto de OpenMantra (el rectángulo rojo) con un
poco de contexto atenuado. Quien etiqueta decide cómo es el fondo DETRÁS de las
letras, dentro del rectángulo: es lo que habría que reconstruir al borrarlas.
Lo que esté fuera del rectángulo no cuenta.

    1 = Uniforme      papel blanco o gris parejo detrás de las letras.
    2 = Estructurado  trama de puntos, rayas o degradado detrás de las letras.
    3 = Complejo      dibujo (personajes, objetos, sombras) detrás de las letras.
    0 = Descartar     no es texto o el rectángulo está mal puesto.
    Retroceso = deshacer la última · Esc = salir (lo etiquetado ya quedó guardado).

Casi todo el texto de OpenMantra está en globos blancos, así que las regiones
con fondo cargado se muestran más seguido (ver muestreo.py); la evaluación
corrige ese sesgo con pesos. Todas las personas ven las regiones en el mismo
orden. Cada etiqueta se guarda al instante en evaluacion/etiquetas.csv.

Uso, desde la raíz del repositorio:
    .venv\\Scripts\\python -m evaluacion.etiquetar --nombre Tomas
    .venv\\Scripts\\python -m evaluacion.etiquetar --nombre Tomas --repasar
Con --repasar se vuelven a mostrar las regiones que esa persona marcó como no
lisas, para confirmarlas o corregirlas.
"""

from __future__ import annotations

import argparse
import csv
import json
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

from . import muestreo

CARPETA = Path(__file__).resolve().parent
DATASET = CARPETA.parent / 'datasets' / 'open-mantra'
ETIQUETAS = CARPETA / 'etiquetas.csv'
CAMPOS = ['libro', 'pagina', 'texto', 'etiqueta', 'etiquetador']
TECLAS = {'1': 'Uniforme', '2': 'Estructurado', '3': 'Complejo', '0': 'Descartada'}
META_POR_CLASE = 15
LADO_VISTA = 480

PREGUNTA = (
    '¿Qué hay DETRÁS de las letras, dentro del rectángulo rojo?\n'
    'Es lo que habría que reconstruir al borrarlas; lo de afuera no cuenta.\n\n'
    '1 = Uniforme: papel liso, blanco o gris parejo\n'
    '2 = Estructurado: trama de puntos, rayas o degradado\n'
    '3 = Complejo: dibujo (personajes, objetos, sombras)\n'
    '0 = Descartar: no es texto o el rectángulo está mal\n\n'
    'Retroceso = deshacer · Esc = salir'
)


def regiones() -> dict[muestreo.Clave, dict]:
    """Todas las cajas de texto de OpenMantra, por clave (libro, página, caja)."""
    datos = json.loads((DATASET / 'annotation.json').read_text(encoding='utf-8'))
    return {
        (libro['book_title'], pagina['page_index'], k): {
            'libro': libro['book_title'],
            'pagina': pagina['page_index'],
            'texto': k,
            'ruta': DATASET / pagina['image_paths']['ja'],
            'caja': (t['x'], t['y'], t['x'] + t['w'], t['y'] + t['h']),
        }
        for libro in datos
        for pagina in libro['pages']
        for k, t in enumerate(pagina['text'])
    }


def leer() -> list[dict]:
    """Filas ya guardadas en etiquetas.csv."""
    if not ETIQUETAS.exists():
        return []
    with open(ETIQUETAS, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def escribir(filas: list[dict]) -> None:
    """Reescribe etiquetas.csv completo."""
    with open(ETIQUETAS, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)


def clave(fila: dict) -> muestreo.Clave:
    """Identifica una región: (libro, página, número de caja)."""
    return fila['libro'], int(fila['pagina']), int(fila['texto'])


def vista_region(region: dict) -> Image.Image:
    """Recorte de la página alrededor de la caja, con el exterior atenuado."""
    with Image.open(region['ruta']) as pagina:
        pagina = pagina.convert('RGB')
        x1, y1, x2, y2 = region['caja']
        m = max(x2 - x1, y2 - y1) * 0.6
        zona = (
            max(0, int(x1 - m)),
            max(0, int(y1 - m)),
            min(pagina.width, int(x2 + m)),
            min(pagina.height, int(y2 + m)),
        )
        recorte = pagina.crop(zona)
    caja = (x1 - zona[0], y1 - zona[1], x2 - zona[0], y2 - zona[1])
    # Atenuar lo de afuera para que la atención quede dentro del rectángulo.
    atenuado = Image.blend(recorte, Image.new('RGB', recorte.size, 'white'), 0.55)
    atenuado.paste(recorte.crop(caja), caja[:2])
    ImageDraw.Draw(atenuado).rectangle(caja, outline=(230, 30, 30), width=2)
    atenuado.thumbnail((LADO_VISTA, LADO_VISTA))
    escala = LADO_VISTA / max(atenuado.size)
    if escala > 1:
        nuevo = (int(atenuado.width * escala), int(atenuado.height * escala))
        atenuado = atenuado.resize(nuevo, Image.LANCZOS)
    return atenuado


class Etiquetador:
    """Ventana que muestra una región y guarda la etiqueta elegida con el teclado."""

    def __init__(self, root: tk.Tk, nombre: str, repasar: bool = False):
        """Prepara la lista de regiones pendientes de esta persona y la interfaz."""
        self.root, self.nombre, self.repasar = root, nombre, repasar
        todas = regiones()
        propias = {
            clave(f): f['etiqueta'] for f in leer() if f['etiquetador'] == nombre
        }
        if repasar:
            claves = [
                c
                for c in muestreo.ordenar(list(todas))
                if propias.get(c, 'Uniforme') != 'Uniforme'
            ]
        else:
            claves = [c for c in muestreo.ordenar(list(todas)) if c not in propias]
        self.pendientes = [todas[c] for c in claves]
        # Cada entrada: (región, fila anterior de esta persona o None), para deshacer.
        self.historial: list[tuple[dict, dict | None]] = []
        modo = ' · repaso' if repasar else ''
        root.title(f'PanelClean · Etiquetar fondos · {nombre}{modo}')
        self.imagen = tk.Label(root)
        # Imagen a la izquierda y texto a la derecha: así cabe en pantallas de 768 px de alto.
        self.imagen.pack(side='left', padx=10, pady=10)
        self.info = tk.Label(
            root, justify='left', anchor='nw', font=('Segoe UI', 11), wraplength=380
        )
        self.info.pack(side='left', fill='y', padx=(0, 14), pady=10)
        root.bind('<Key>', self.tecla)
        self.mostrar()

    def propias(self) -> dict[muestreo.Clave, dict]:
        """Filas de esta persona, por región."""
        return {clave(f): f for f in leer() if f['etiquetador'] == self.nombre}

    def mostrar(self) -> None:
        """Dibuja la región actual y el estado del etiquetado."""
        if not self.pendientes:
            fin = 'Repaso terminado.' if self.repasar else 'No quedan regiones.'
            self.info.config(text=f'{fin} ¡Gracias! Ya puedes cerrar la ventana.')
            return
        r = self.pendientes[0]
        self.foto = ImageTk.PhotoImage(vista_region(r))
        self.imagen.config(image=self.foto)
        propias = self.propias()
        cuenta = {c: 0 for c in TECLAS.values()}
        for f in propias.values():
            cuenta[f['etiqueta']] += 1
        progreso = '\n'.join(
            f'  {c}: {cuenta[c]} de {META_POR_CLASE}'
            for c in ('Uniforme', 'Estructurado', 'Complejo')
        )
        antes = propias.get(clave(r))
        nota = f'\n\nAntes la marcaste como: {antes["etiqueta"]}' if antes else ''
        self.info.config(
            text=(
                f'{PREGUNTA}\n\n'
                f'Llevas:\n{progreso}\n  Descartadas: {cuenta["Descartada"]}\n\n'
                f'Región: {r["libro"]}, página {r["pagina"]}, caja {r["texto"]}{nota}'
            )
        )

    def tecla(self, evento: tk.Event) -> None:
        """Guarda la etiqueta de la tecla pulsada, deshace o sale."""
        if evento.keysym == 'Escape':
            self.root.destroy()
        elif evento.keysym == 'BackSpace' and self.historial:
            region, anterior = self.historial.pop()
            filas = [f for f in leer() if not self._es_mia(f, region)]
            if anterior is not None:
                filas.append(anterior)
            escribir(filas)
            self.pendientes.insert(0, region)
            self.mostrar()
        elif evento.char in TECLAS and self.pendientes:
            region = self.pendientes.pop(0)
            filas = leer()
            anterior = next((f for f in filas if self._es_mia(f, region)), None)
            filas = [f for f in filas if not self._es_mia(f, region)]
            filas.append(
                {
                    'libro': region['libro'],
                    'pagina': region['pagina'],
                    'texto': region['texto'],
                    'etiqueta': TECLAS[evento.char],
                    'etiquetador': self.nombre,
                }
            )
            escribir(filas)
            self.historial.append((region, anterior))
            self.mostrar()

    def _es_mia(self, fila: dict, region: dict) -> bool:
        """Indica si la fila es la etiqueta de esta persona para esa región."""
        return fila['etiquetador'] == self.nombre and clave(fila) == clave(region)


def main() -> None:
    """Abre la ventana de etiquetado para la persona indicada."""
    parser = argparse.ArgumentParser(description='Etiquetar fondos de OpenMantra.')
    parser.add_argument('--nombre', required=True, help='quién etiqueta')
    parser.add_argument(
        '--repasar',
        action='store_true',
        help='volver a mostrar las regiones que marcaste como no lisas',
    )
    args = parser.parse_args()
    root = tk.Tk()
    Etiquetador(root, args.nombre.strip(), args.repasar)
    root.mainloop()


if __name__ == '__main__':
    main()
