"""Programa para marcar dónde hay texto en páginas propias (tarea T1.2.7).

Sirve para enseñarle al detector otros estilos e idiomas y para medir cómo le va en
ellos. Se abre una carpeta con las páginas de UNA serie (imágenes o PDF), o un PDF, y
en cada página se marca un rectángulo por bloque de texto. El detector propone los que
encuentra, para partir de ahí: se corrigen, se borran o se agregan.

Cómo marcar:
- Un rectángulo por bloque de texto, ajustado a las letras (no al globo).
- TODO el texto que habría que borrar: lo que quede sin marcar en una página lista
  se aprende como «no texto».
- Onomatopeyas y letreros que son parte del dibujo: marcarlos como «ignorar».

Una serie «reservada» se usa solo para medir, nunca para entrenar. En ella no se
muestran propuestas: si la referencia partiera de lo que el detector encontró, la
medición sería más favorable de lo que corresponde.

Las imágenes no se copian a ningún lado (ver cajas.py).

Uso, desde la raíz del repositorio (o con MARCAR_WINDOWS.bat):
    .venv\\Scripts\\python -m evaluacion.marcar
    .venv\\Scripts\\python -m evaluacion.marcar "C:\\mis\\hojas" --nombre Ana
    .venv\\Scripts\\python -m evaluacion.marcar "C:\\mis\\tomo_01.pdf" --nombre Ana
"""

from __future__ import annotations

import argparse
import copy
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from panelclean import vision as V
from panelclean.app import BotonRedondeado, C, fuente, registrar_fuentes

from . import cajas as K

AYUDA = (
    'Marca TODO el texto que habría que borrar, un rectángulo por bloque, '
    'ajustado a las letras.\n\n'
    'Arrastrar en un lugar vacío: caja nueva\n'
    'Clic en una caja: elegirla (se mueve desde\n'
    'adentro y se ajusta desde sus esquinas)\n'
    'Supr: borrar la caja elegida\n'
    'I: texto / ignorar (onomatopeyas)\n'
    'Ctrl + Z: deshacer\n'
    'Rueda: zoom · clic derecho: mover · 0: ajustar\n'
    'Enter: página lista y siguiente\n'
    'Av Pág / Re Pág: cambiar de página\n'
    'G: ir a la página número…'
)
MINIMO = 6  # px de la imagen: lado mínimo de una caja


class Marcador:
    """Ventana para marcar las cajas de texto de las páginas de una carpeta o un PDF."""

    def __init__(self, root: tk.Tk, carpeta: Path, nombre: str):
        """Prepara la ventana y abre la primera página que no esté lista."""
        self.root, self.nombre = root, nombre
        avisos: list[str] = []  # PDF que no se pudieron leer
        self.paginas = K.paginas_de(carpeta, avisos)
        self.grupo = K.registrar_carpeta(carpeta)
        self.series = K.leer()
        self.serie = self.series.setdefault(
            self.grupo, {'reservada': False, 'paginas': {}}
        )
        self.reservada = tk.BooleanVar(root, value=self.serie['reservada'])
        self.indice = 0
        self.imagen: np.ndarray | None = None
        self.cajas: list[dict] = []
        self.elegida: int | None = None
        self.historial: list[list[dict]] = []
        self.hubo_listas = False  # se terminó alguna página en esta sesión
        self.zoom, self.panx, self.pany = 1.0, 0.0, 0.0
        self._accion = None  # arrastre en curso: (tipo, datos)
        self._vista = (1.0, 0.0, 0.0)  # escala y origen de la página en el lienzo
        self._construir()
        if self.paginas:
            pendientes = [
                i for i, p in enumerate(self.paginas) if not self._lista(p.name)
            ]
            self.abrir(pendientes[0] if pendientes else 0)
        else:
            self.estado.configure(text=f'No hay páginas que abrir en {carpeta}.')
        if avisos:
            messagebox.showwarning(
                'PDF que no se pudieron leer', '\n'.join(avisos), parent=root
            )

    # --- ventana ----------------------------------------------------------------
    def _construir(self) -> None:
        """Lienzo a la izquierda; instrucciones, avance y botones a la derecha."""
        self.root.title(f'PanelClean · Marcar texto · {self.grupo}')
        self.root.configure(bg=C['bg'])
        lado = tk.Frame(self.root, bg=C['panel'], width=300)
        lado.pack(side='right', fill='y')
        lado.pack_propagate(False)
        self.lienzo = tk.Canvas(
            self.root, bg=C['canvas'], highlightthickness=0, cursor='crosshair'
        )
        self.lienzo.pack(side='left', fill='both', expand=True)

        def texto(contenido, tam=9, negrita=False, color='text'):
            """Etiqueta del panel derecho."""
            e = tk.Label(
                lado,
                text=contenido,
                bg=C['panel'],
                fg=C[color],
                font=fuente(tam, negrita),
                justify='left',
                anchor='w',
                wraplength=264,
            )
            e.pack(fill='x', padx=16, pady=(8, 0))
            return e

        texto('Marcar texto', 14, True)
        texto(AYUDA, 9, color='muted')
        self.avance = texto('', 10, True)
        self.estado = texto('', 9, color='accent')
        casilla = tk.Checkbutton(
            lado,
            text='Serie reservada para medir\n(no se entrena con ella)',
            variable=self.reservada,
            command=self.cambiar_reservada,
            bg=C['panel'],
            fg=C['text'],
            selectcolor=C['bg'],
            activebackground=C['panel'],
            font=fuente(9),
            justify='left',
            anchor='w',
            bd=0,
            highlightthickness=0,
        )
        casilla.pack(fill='x', padx=14, pady=(12, 0))
        for nombre, accion, primario in (
            ('Página lista  (Enter)', self.pagina_lista, True),
            ('Texto / ignorar  (I)', self.cambiar_tipo, False),
            ('Borrar caja  (Supr)', self.borrar, False),
            ('Borrar todas', self.borrar_todas, False),
        ):
            BotonRedondeado(lado, nombre, accion, primario).pack(
                fill='x', padx=16, pady=(10, 0)
            )
        cv = self.lienzo
        cv.bind('<ButtonPress-1>', self.clic)
        cv.bind('<B1-Motion>', self.arrastre)
        cv.bind('<ButtonRelease-1>', self.soltar)
        cv.bind('<ButtonPress-3>', self._mover_inicio)
        cv.bind('<B3-Motion>', self._mover)
        cv.bind('<MouseWheel>', lambda e: self._acercar(e, 1.15 ** (e.delta / 120)))
        cv.bind('<Button-4>', lambda e: self._acercar(e, 1.15))
        cv.bind('<Button-5>', lambda e: self._acercar(e, 1 / 1.15))
        cv.bind('<Configure>', lambda _: self.dibujar())
        for tecla, accion in (
            ('<Delete>', self.borrar),
            ('<BackSpace>', self.borrar),
            ('<Return>', self.pagina_lista),
            ('<Next>', lambda: self.cambiar_pagina(1)),
            ('<Prior>', lambda: self.cambiar_pagina(-1)),
            ('<Control-z>', self.deshacer),
            ('i', self.cambiar_tipo),
            ('I', self.cambiar_tipo),
            ('0', self.ajustar_vista),
            ('g', self.ir_a_pagina),
            ('G', self.ir_a_pagina),
            ('<Escape>', lambda: self.elegir(None)),
        ):
            self.root.bind(tecla, lambda _, a=accion: a())
        self.root.protocol('WM_DELETE_WINDOW', self.cerrar)

    # --- páginas ------------------------------------------------------------------
    def _lista(self, archivo: str) -> bool:
        """Indica si la página ya se marcó como lista."""
        return bool(self.serie['paginas'].get(archivo, {}).get('completa'))

    def abrir(self, indice: int) -> None:
        """Muestra la página indice: lo ya marcado en ella o, si es nueva, lo que
        propone el detector (salvo en una serie reservada)."""
        self.indice = indice
        ruta = self.paginas[indice]
        self.estado.configure(text='Abriendo…')
        self.root.update_idletasks()
        try:
            self.imagen = V.bgr_from_file(str(ruta))
        except Exception as exc:
            self.imagen, self.cajas = None, []
            self.estado.configure(text=f'No se pudo abrir {ruta.name}: {exc}')
            self.dibujar()
            return
        guardada = self.serie['paginas'].get(ruta.name)
        if guardada is not None:
            self.cajas = copy.deepcopy(guardada['cajas'])
            aviso = 'Lista.' if guardada['completa'] else 'Sin terminar.'
            if self.reservada.get() and not guardada['completa']:
                self.cajas = [c for c in self.cajas if c['origen'] != 'propuesta']
        elif self.reservada.get():
            self.cajas, aviso = [], 'Serie reservada: sin propuestas.'
        else:
            self.cajas = [
                {
                    'x': r.x,
                    'y': r.y,
                    'w': r.w,
                    'h': r.h,
                    'tipo': 'texto',
                    'origen': 'propuesta',
                }
                for r in V.detectar_texto(self.imagen)
            ]
            aviso = f'{len(self.cajas)} cajas propuestas (punteadas): revísalas.'
        self.historial, self.elegida = [], None
        self.zoom, self.panx, self.pany = 1.0, 0.0, 0.0
        self.estado.configure(text=f'{ruta.name}\n{aviso}')
        self._actualizar_avance()
        self.dibujar()

    def _actualizar_avance(self) -> None:
        """Texto con la página actual y cuántas van listas."""
        listas = sum(self._lista(p.name) for p in self.paginas)
        self.avance.configure(
            text=f'Página {self.indice + 1} de {len(self.paginas)}  ·  {listas} listas'
        )

    def _guardar(self, completa: bool) -> None:
        """Guarda las cajas de la página abierta. Una propuesta en una página lista
        queda igual, con su origen: se puede saber cuánto partió del detector."""
        if self.imagen is None:
            return
        alto, ancho = self.imagen.shape[:2]
        self.serie['paginas'][self.paginas[self.indice].name] = {
            'ancho': ancho,
            'alto': alto,
            'completa': completa,
            'etiquetador': self.nombre,
            'cajas': copy.deepcopy(self.cajas),
        }
        self.serie['reservada'] = bool(self.reservada.get())
        K.guardar(self.series)

    def _guardar_borrador(self) -> None:
        """Guarda los cambios de una página sin terminar (o ya lista) al dejarla."""
        if self.historial:
            self._guardar(self._lista(self.paginas[self.indice].name))

    def pagina_lista(self) -> None:
        """Confirma que todo el texto de la página está marcado y pasa a la siguiente."""
        if self.imagen is None:
            return
        self._guardar(True)
        self.hubo_listas = True
        pendientes = [
            i
            for i in list(range(self.indice + 1, len(self.paginas)))
            + list(range(self.indice))
            if not self._lista(self.paginas[i].name)
        ]
        if pendientes:
            self.abrir(pendientes[0])
        else:
            self._actualizar_avance()
            self.estado.configure(
                text='Todas las páginas están listas. Cierra la ventana para entrenar.'
            )

    def cambiar_pagina(self, paso: int) -> None:
        """Pasa a otra página; lo marcado en la actual se guarda como borrador."""
        destino = self.indice + paso
        if 0 <= destino < len(self.paginas):
            self._guardar_borrador()
            self.abrir(destino)

    def ir_a_pagina(self) -> None:
        """Pide un número de página y la abre (en un PDF puede haber cientos)."""
        if not self.paginas:
            return
        numero = simpledialog.askinteger(
            'Ir a una página',
            f'Número de página (1 a {len(self.paginas)}):',
            parent=self.root,
            initialvalue=self.indice + 1,
            minvalue=1,
            maxvalue=len(self.paginas),
        )
        if numero and numero - 1 != self.indice:
            self.cambiar_pagina(numero - 1 - self.indice)

    def cambiar_reservada(self) -> None:
        """Marca o desmarca la serie como reservada. Al reservarla, la página abierta
        pierde las propuestas que nadie revisó, y si ya tenía páginas listas se avisa
        que partieron de propuestas o que pudieron usarse para entrenar."""
        reservada = self.reservada.get()
        if reservada and K.completas(self.serie):
            if not messagebox.askyesno(
                'Reservar la serie',
                'Esta serie ya tiene páginas listas. Si se usaron para entrenar, o '
                'partieron de las propuestas del detector, medir con ellas dará un '
                'resultado más favorable que el real.\n\n¿Reservarla de todas formas?',
            ):
                self.reservada.set(False)
                return
        self.serie['reservada'] = bool(reservada)
        if self.serie['paginas']:
            K.guardar(self.series)
        if (
            reservada
            and self.paginas
            and not self._lista(self.paginas[self.indice].name)
        ):
            self._recordar()
            self.cajas = [c for c in self.cajas if c['origen'] != 'propuesta']
            self.elegir(None)

    # --- cajas --------------------------------------------------------------------
    def _recordar(self) -> None:
        """Guarda el estado de las cajas para «Deshacer»."""
        self.historial.append(copy.deepcopy(self.cajas))
        self.historial = self.historial[-50:]

    def deshacer(self) -> None:
        """Vuelve al estado anterior de las cajas."""
        if self.historial:
            self.cajas = self.historial.pop()
            self.elegir(None)

    def elegir(self, indice: int | None) -> None:
        """Elige una caja (o ninguna)."""
        self.elegida = indice
        self.dibujar()

    def borrar(self) -> None:
        """Borra la caja elegida."""
        if self.elegida is not None:
            self._recordar()
            del self.cajas[self.elegida]
            self.elegir(None)

    def borrar_todas(self) -> None:
        """Borra todas las cajas de la página (por ejemplo, propuestas malas)."""
        if self.cajas:
            self._recordar()
            self.cajas = []
            self.elegir(None)

    def cambiar_tipo(self) -> None:
        """Alterna la caja elegida entre texto e ignorar."""
        if self.elegida is not None:
            self._recordar()
            c = self.cajas[self.elegida]
            c['tipo'] = 'ignorar' if c['tipo'] == 'texto' else 'texto'
            c['origen'] = 'mano'
            self.dibujar()

    def a_imagen(self, x: float, y: float) -> tuple[int, int]:
        """Punto del lienzo en píxeles de la página, sin salirse de ella."""
        escala, ox, oy = self._vista
        alto, ancho = self.imagen.shape[:2]
        return (
            int(min(ancho, max(0, round((x - ox) / escala)))),
            int(min(alto, max(0, round((y - oy) / escala)))),
        )

    def _esquina(self, x: float, y: float) -> int | None:
        """Esquina de la caja elegida bajo el cursor (0 a 3), o None."""
        if self.elegida is None:
            return None
        escala, ox, oy = self._vista
        c = self.cajas[self.elegida]
        for k in range(4):
            ex = ox + (c['x'] + (k % 2) * c['w']) * escala
            ey = oy + (c['y'] + (k // 2) * c['h']) * escala
            if abs(ex - x) <= 8 and abs(ey - y) <= 8:
                return k
        return None

    def clic(self, evento) -> None:
        """Empieza a ajustar una esquina, a mover una caja o a dibujar una nueva."""
        if self.imagen is None:
            return
        self.lienzo.focus_set()
        punto = self.a_imagen(evento.x, evento.y)
        esquina = self._esquina(evento.x, evento.y)
        if esquina is not None:
            self._recordar()
            c = self.cajas[self.elegida]
            fijo = (
                c['x'] + (1 - esquina % 2) * c['w'],
                c['y'] + (1 - esquina // 2) * c['h'],
            )
            self._accion = ('esquina', fijo)
            return
        # La caja más chica bajo el cursor: una caja grande no tapa a las de adentro.
        bajo = [
            (c['w'] * c['h'], k)
            for k, c in enumerate(self.cajas)
            if c['x'] <= punto[0] <= c['x'] + c['w']
            and c['y'] <= punto[1] <= c['y'] + c['h']
        ]
        if bajo:
            self.elegir(min(bajo)[1])
            c = self.cajas[self.elegida]
            self._accion = ('mover', (punto, (c['x'], c['y']), False))
        else:
            self.elegir(None)
            self._accion = ('nueva', punto)

    def arrastre(self, evento) -> None:
        """Sigue la acción que empezó con el clic."""
        if self._accion is None:
            return
        tipo, datos = self._accion
        x, y = self.a_imagen(evento.x, evento.y)
        if tipo == 'mover':
            inicio, origen, movida = datos
            if not movida and abs(x - inicio[0]) + abs(y - inicio[1]) < 3:
                return
            if not movida:
                self._recordar()
            c = self.cajas[self.elegida]
            alto, ancho = self.imagen.shape[:2]
            c['x'] = int(min(ancho - c['w'], max(0, origen[0] + x - inicio[0])))
            c['y'] = int(min(alto - c['h'], max(0, origen[1] + y - inicio[1])))
            c['origen'] = 'mano'
            self._accion = ('mover', (inicio, origen, True))
            self.dibujar()
        elif tipo == 'esquina':
            self._poner(self.cajas[self.elegida], datos, (x, y))
            self.dibujar()
        else:
            escala, ox, oy = self._vista
            self.lienzo.delete('nueva')
            self.lienzo.create_rectangle(
                ox + datos[0] * escala,
                oy + datos[1] * escala,
                ox + x * escala,
                oy + y * escala,
                outline=C['accent'],
                width=2,
                tags='nueva',
            )

    @staticmethod
    def _poner(caja: dict, a: tuple, b: tuple) -> None:
        """Fija el rectángulo de la caja a partir de dos esquinas opuestas."""
        caja['x'], caja['y'] = int(min(a[0], b[0])), int(min(a[1], b[1]))
        caja['w'] = max(MINIMO, int(abs(a[0] - b[0])))
        caja['h'] = max(MINIMO, int(abs(a[1] - b[1])))
        caja['origen'] = 'mano'

    def soltar(self, evento) -> None:
        """Termina la acción; si era una caja nueva de tamaño suficiente, la agrega."""
        if self._accion is None:
            return
        tipo, datos = self._accion
        self._accion = None
        self.lienzo.delete('nueva')
        if tipo != 'nueva':
            return
        x, y = self.a_imagen(evento.x, evento.y)
        if abs(x - datos[0]) >= MINIMO and abs(y - datos[1]) >= MINIMO:
            self._recordar()
            caja = {'tipo': 'texto'}
            self._poner(caja, datos, (x, y))
            self.cajas.append(caja)
            self.elegir(len(self.cajas) - 1)

    # --- vista ----------------------------------------------------------------------
    def ajustar_vista(self) -> None:
        """Vuelve a mostrar la página completa."""
        self.zoom, self.panx, self.pany = 1.0, 0.0, 0.0
        self.dibujar()

    def _acercar(self, evento, factor: float) -> None:
        """Zoom con la rueda, hacia el punto bajo el cursor."""
        if self.imagen is None:
            return
        escala, ox, oy = self._vista
        nuevo = min(12.0, max(0.5, self.zoom * factor))
        px, py = (evento.x - ox) / escala, (evento.y - oy) / escala
        despues = escala * nuevo / self.zoom
        self.zoom = nuevo
        alto, ancho = self.imagen.shape[:2]
        self.panx = evento.x - px * despues - (self._ancho() - ancho * despues) / 2
        self.pany = evento.y - py * despues - (self._alto() - alto * despues) / 2
        self.dibujar()

    def _mover_inicio(self, evento) -> None:
        """Guarda dónde empieza el desplazamiento con el botón derecho."""
        self._ancla = (evento.x, evento.y)

    def _mover(self, evento) -> None:
        """Desplaza la página con el botón derecho."""
        self.panx += evento.x - self._ancla[0]
        self.pany += evento.y - self._ancla[1]
        self._ancla = (evento.x, evento.y)
        self.dibujar()

    def _ancho(self) -> int:
        """Ancho del lienzo en pantalla."""
        return max(100, self.lienzo.winfo_width())

    def _alto(self) -> int:
        """Alto del lienzo en pantalla."""
        return max(100, self.lienzo.winfo_height())

    def dibujar(self) -> None:
        """Dibuja la parte visible de la página y sus cajas."""
        cv = self.lienzo
        cv.delete('all')
        if self.imagen is None:
            return
        alto, ancho = self.imagen.shape[:2]
        w, h = self._ancho(), self._alto()
        escala = min((w - 20) / ancho, (h - 20) / alto) * self.zoom
        ox = (w - ancho * escala) / 2 + self.panx
        oy = (h - alto * escala) / 2 + self.pany
        self._vista = (escala, ox, oy)
        x0, y0 = max(0, int(-ox / escala)), max(0, int(-oy / escala))
        x1 = min(ancho, int(np.ceil((w - ox) / escala)))
        y1 = min(alto, int(np.ceil((h - oy) / escala)))
        if x1 > x0 and y1 > y0:
            tamano = (
                max(1, round((x1 - x0) * escala)),
                max(1, round((y1 - y0) * escala)),
            )
            parte = cv2.resize(
                self.imagen[y0:y1, x0:x1],
                tamano,
                interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC,
            )
            self._foto = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(parte, cv2.COLOR_BGR2RGB)),
                master=self.root,
            )
            cv.create_image(
                round(ox + x0 * escala),
                round(oy + y0 * escala),
                image=self._foto,
                anchor='nw',
            )
        for k, c in enumerate(self.cajas):
            elegida = k == self.elegida
            color = C['muted'] if c['tipo'] == 'ignorar' else C['accent']
            a, b = ox + c['x'] * escala, oy + c['y'] * escala
            d, e = a + c['w'] * escala, b + c['h'] * escala
            opciones = {'dash': (5, 3)} if c['origen'] == 'propuesta' else {}
            cv.create_rectangle(
                a, b, d, e, outline=color, width=3 if elegida else 2, **opciones
            )
            if c['tipo'] == 'ignorar':
                cv.create_line(a, b, d, e, fill=color)
            if elegida:
                for px, py in ((a, b), (d, b), (a, e), (d, e)):
                    cv.create_rectangle(
                        px - 4, py - 4, px + 4, py + 4, fill=color, outline=C['bg']
                    )

    def cerrar(self) -> None:
        """Guarda lo pendiente y cierra."""
        if self.paginas:
            self._guardar_borrador()
        self.root.destroy()


def pedir_origen(root: tk.Tk) -> str:
    """Pide la carpeta o el PDF con las páginas de una serie ('' si se cancela)."""
    es_pdf = messagebox.askyesnocancel(
        'Marcar texto',
        '¿Las páginas están en un solo archivo PDF?\n\n'
        'Sí: elegir el PDF.\n'
        'No: elegir la carpeta con las imágenes o los PDF de UNA serie.',
        parent=root,
    )
    if es_pdf is None:
        return ''
    if es_pdf:
        return filedialog.askopenfilename(
            title='PDF con las páginas', filetypes=[('PDF', '*.pdf')]
        )
    return filedialog.askdirectory(title='Carpeta con las páginas de UNA serie')


def main() -> None:
    """Pide lo que falte (páginas y nombre), abre la ventana y ofrece entrenar al cerrar."""
    parser = argparse.ArgumentParser(description='Marcar texto en páginas propias.')
    parser.add_argument('carpeta', nargs='?', help='carpeta o PDF de UNA serie')
    parser.add_argument('--nombre', help='quién marca')
    args = parser.parse_args()
    registrar_fuentes()
    root = tk.Tk()
    root.withdraw()
    carpeta = args.carpeta or pedir_origen(root)
    if not carpeta:
        return
    nombre = args.nombre or simpledialog.askstring(
        'Marcar texto', '¿Quién marca? (tu nombre)', parent=root
    )
    if not nombre:
        return
    root.geometry(
        f'{min(1280, root.winfo_screenwidth() - 40)}x'
        f'{min(860, root.winfo_screenheight() - 90)}+10+10'
    )
    root.deiconify()
    ventana = Marcador(root, Path(carpeta), nombre.strip())
    root.mainloop()
    if ventana.hubo_listas:
        from . import entrenar_detector

        pregunta = tk.Tk()
        pregunta.withdraw()
        entrenar = messagebox.askyesno(
            'Entrenar el detector',
            '¿Entrenar el detector ahora con lo marcado?\n\n'
            'Tarda uno o dos minutos. También se puede hacer después con\n'
            'python -m evaluacion.entrenar_detector',
            parent=pregunta,
        )
        pregunta.destroy()
        if entrenar:
            entrenar_detector.main_con([])


if __name__ == '__main__':
    main()
