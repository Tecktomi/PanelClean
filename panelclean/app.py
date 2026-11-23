"""PanelClean 0.3: editor local de limpieza asistida de globos de manga."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
import shutil
import sys
from functools import lru_cache
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import __version__
from .vision import (
    Region,
    Options,
    bgr_from_file,
    write_png,
    bounded_region,
    detect_bubbles,
    iou_rect,
    texto_en_zona,
    generate_mask,
    describe_background,
    clean_region,
    mask_whole,
    inner_area,
)

ROOT = Path(__file__).resolve().parent.parent
FUENTES = Path(__file__).resolve().parent / 'fuentes'

# Paleta «Sakura» clara: fondos rosa pálido, acento rosa cerezo y texto ciruela.
C = {
    'bg': '#fff7f9',  # fondo de la ventana y de las tarjetas
    'top': '#fbe6ee',  # barras superior e inferior
    'panel': '#fcecf1',  # paneles laterales
    'panel2': '#f6d6e1',  # botones y lista de regiones
    'hover': '#f0c3d3',  # botón bajo el cursor
    'line': '#ecc5d3',  # bordes
    'text': '#4a2c3a',  # texto principal (ciruela)
    'muted': '#9b7383',  # texto secundario (malva)
    'accent': '#e36b93',  # rosa cerezo: acción principal y región elegida
    'accent_hover': '#d4567f',
    'on_accent': '#ffffff',  # texto sobre el rosa cerezo
    'region': '#a9718a',  # rectángulo de una región no elegida
    'warn': '#d9822b',  # aviso: región para revisar
    'canvas': '#f6dde6',  # fondo detrás de la página
    'select': '#f3bfd1',  # fila elegida en la lista
}
# Quicksand viene en panelclean/fuentes (licencia OFL) y se carga al iniciar;
# si no se pudo cargar, Tk usa su fuente por defecto en su lugar.
FAMILIA = 'Quicksand'
# Las máscaras se ven en rosa semitransparente sobre la vista ORIGINAL (alfa ≈ 40 %).
ROSA_MASCARA = np.array([255, 71, 100], np.float32)
ALFA_MASCARA = 104 / 255


def fuente(tam: int = 10, negrita: bool = False) -> tuple:
    """Fuente de la interfaz en el tamaño pedido."""
    return (FAMILIA, tam, 'bold' if negrita else 'normal')


FONT = fuente(10)
SMALL = fuente(9)


def registrar_fuentes() -> None:
    """Hace disponible Quicksand sin instalarla en el sistema.

    En Windows se registra solo para este proceso; en Linux se copia a la
    carpeta de fuentes del usuario si aún no está. Si algo falla, la interfaz
    sigue funcionando con la fuente por defecto de Tk.
    """
    archivos = sorted(FUENTES.glob('*.ttf'))
    try:
        if sys.platform == 'win32':
            import ctypes

            for archivo in archivos:
                ctypes.windll.gdi32.AddFontResourceExW(str(archivo), 0x10, 0)
        elif sys.platform.startswith('linux'):
            destino = Path.home() / '.local' / 'share' / 'fonts' / 'PanelClean'
            destino.mkdir(parents=True, exist_ok=True)
            for archivo in archivos:
                if not (destino / archivo.name).exists():
                    shutil.copy2(archivo, destino / archivo.name)
    except (OSError, AttributeError):
        pass


@lru_cache(maxsize=256)
def _rectangulo_redondeado(
    ancho: int, alto: int, radio: int, color: str, fondo: str
) -> Image.Image:
    """Rectángulo con esquinas redondeadas y bordes suaves (se dibuja al triple y se reduce)."""
    k = 3
    grande = Image.new('RGB', (ancho * k, alto * k), fondo)
    ImageDraw.Draw(grande).rounded_rectangle(
        (0, 0, ancho * k - 1, alto * k - 1), radius=radio * k, fill=color
    )
    return grande.resize((ancho, alto), Image.Resampling.LANCZOS)


class BotonRedondeado(tk.Canvas):
    """Botón con las esquinas redondeadas, que Tk no ofrece.

    El fondo es una imagen hecha con Pillow y el texto lo dibuja el propio
    lienzo. Admite configure(text=...) para cambiar el texto.
    """

    def __init__(
        self,
        parent,
        text: str,
        command,
        primario: bool = False,
        alto: int = 34,
        ancho: int | None = None,
        tam: int = 10,
    ):
        self._fondo = parent.cget('bg')
        self._fuente = fuente(tam, True)
        if ancho is None:
            ancho = tkfont.Font(root=parent, font=self._fuente).measure(text) + 30
        super().__init__(
            parent,
            width=ancho,
            height=alto,
            bg=self._fondo,
            highlightthickness=0,
            bd=0,
            cursor='hand2',
        )
        self._texto = text
        self._comando = command
        if primario:
            self._colores = (C['accent'], C['accent_hover'], C['on_accent'])
        else:
            self._colores = (C['panel2'], C['hover'], C['text'])
        self._bajo_cursor = False
        self._imagen = None
        self.bind('<Configure>', lambda _: self._dibujar())
        self.bind('<Enter>', lambda _: self._marcar(True))
        self.bind('<Leave>', lambda _: self._marcar(False))
        self.bind('<ButtonRelease-1>', self._soltar)

    def _marcar(self, bajo_cursor: bool) -> None:
        """Cambia el color mientras el cursor está encima."""
        self._bajo_cursor = bajo_cursor
        self._dibujar()

    def _soltar(self, event) -> None:
        """Ejecuta la acción si se suelta el clic dentro del botón."""
        if 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height():
            self._comando()

    def _dibujar(self) -> None:
        """Dibuja el fondo redondeado y el texto con el tamaño actual."""
        ancho, alto = self.winfo_width(), self.winfo_height()
        if ancho < 4 or alto < 4:
            return
        normal, encima, color_texto = self._colores
        color = encima if self._bajo_cursor else normal
        radio = min(10, alto // 2)
        self._imagen = ImageTk.PhotoImage(
            _rectangulo_redondeado(ancho, alto, radio, color, self._fondo), master=self
        )
        self.delete('all')
        self.create_image(0, 0, image=self._imagen, anchor='nw')
        self.create_text(
            ancho / 2, alto / 2, text=self._texto, fill=color_texto, font=self._fuente
        )

    def configure(self, cnf=None, **kw):
        """Como Canvas.configure, pero además acepta text=..."""
        if 'text' in kw:
            self._texto = kw.pop('text')
            self._dibujar()
        if cnf or kw:
            return super().configure(cnf, **kw)
        return None

    config = configure


class Tooltip:
    """Ayuda contextual sin ocupar espacio fijo de la pantalla."""

    def __init__(self, widget, content):
        """Asocia el texto de ayuda al widget; aparece al dejar el cursor encima."""
        self.widget = widget
        self.content = content
        self.popup = None
        self.pending = None
        widget.bind('<Enter>', self.schedule, add='+')
        widget.bind('<Leave>', self.close, add='+')
        widget.bind('<Button>', self.close, add='+')

    def schedule(self, event=None):
        """Programa la ayuda para que aparezca tras una breve pausa del cursor."""
        self.cancel()
        self.pending = self.widget.after(650, self.open)

    def cancel(self):
        """Cancela la aparición programada, si la hay."""
        if self.pending is not None:
            self.widget.after_cancel(self.pending)
            self.pending = None

    def open(self):
        """Muestra la ayuda en una ventanita sin bordes bajo el widget."""
        self.pending = None
        if self.popup or not self.widget.winfo_exists():
            return
        self.popup = tk.Toplevel(self.widget)
        self.popup.wm_overrideredirect(True)
        self.popup.configure(bg=C['line'])
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.popup.geometry(f'+{x}+{y}')
        tk.Label(
            self.popup,
            text=self.content,
            bg=C['bg'],
            fg=C['text'],
            font=SMALL,
            justify='left',
            wraplength=280,
            padx=11,
            pady=8,
        ).pack(padx=1, pady=1)

    def close(self, event=None):
        """Oculta la ayuda y cancela cualquier aparición pendiente."""
        self.cancel()
        if self.popup:
            self.popup.destroy()
            self.popup = None


# Datos de la hoja abierta que cambian al pasar a otra hoja de la cola (ver Hoja).
CAMPOS_HOJA = (
    'original',
    'working',
    'regions',
    'applied_ids',
    'undo_stack',
    'next_id',
    'selected_indices',
    'focus_index',
    'zoom',
    'panx',
    'pany',
)
PASOS_AL_SALIR = 3  # pasos de «Deshacer» que conserva una hoja que no está abierta
ALTO_MINIATURA = 62  # px, en la tira de hojas


@dataclass
class Hoja:
    """Una página de la cola, con todo lo que se edita en ella.

    La hoja abierta se edita directamente en los atributos de PanelCleanApp con los
    mismos nombres (CAMPOS_HOJA); al pasar a otra hoja se guardan aquí.
    procesada: se limpió al menos una vez (a mano o con «Procesar todas»).
    exportada: se exportó y no cambió después.
    """

    ruta: str
    original: np.ndarray
    working: np.ndarray
    regions: list = field(default_factory=list)
    applied_ids: set = field(default_factory=set)
    undo_stack: list = field(default_factory=list)
    next_id: int = 1
    selected_indices: set = field(default_factory=set)
    focus_index: int | None = None
    zoom: float = 1.0
    panx: float = 0.0
    pany: float = 0.0
    procesada: bool = False
    exportada: bool = False

    @property
    def nombre(self) -> str:
        """Nombre del archivo de la página."""
        return Path(self.ruta).name

    @property
    def estado(self) -> str:
        """'Pendiente' (sin limpiar), 'Lista' (todas sus regiones limpias) o 'Revisar'
        (quedan regiones sin limpiar, o no se encontró texto)."""
        if not self.procesada:
            return 'Pendiente'
        if self.regions and all(r.id in self.applied_ids for r in self.regions):
            return 'Lista'
        return 'Revisar'


COLOR_ESTADO = {'Lista': 'accent', 'Revisar': 'warn', 'Pendiente': 'muted'}
TIPOS_IMAGEN = [('Imágenes', '*.png *.jpg *.jpeg *.bmp *.webp'), ('Todos', '*.*')]


class PanelCleanApp:
    """Ventana principal de PanelClean.

    Estado principal:
    - original: página tal como se abrió (nunca se modifica).
    - working: resultado que se muestra a la derecha; se recompone desde original.
    - regions: rectángulos marcados, cada uno con su máscara. Cada región tiene
      un id interno que nunca se repite (lo usa «Deshacer») y un número visible
      (R01, R02…) que reutiliza el menor número libre.
    - applied_ids: ids de las regiones cuya limpieza está aplicada en working.
    - undo_stack: pasos anteriores para «Deshacer».
    - hojas: la cola de páginas abiertas; actual es el índice de la que se edita.
      El estado de arriba es el de la hoja actual (ver Hoja y CAMPOS_HOJA).
    """

    def __init__(self, root: tk.Tk):
        """Crea el estado vacío y construye la interfaz; empieza sin ninguna página abierta."""
        self.root = root
        root.title(f'PanelClean  |  {__version__}')
        self._fit_window(1480, 865)
        root.configure(bg=C['bg'])
        self.original: np.ndarray | None = None
        self.working: np.ndarray | None = None
        self.regions: list[Region] = []
        self.selected_indices: set[int] = set()
        self.focus_index: int | None = None
        self.applied_ids: set[int] = set()
        # Cada paso guarda (imagen resultado, ids aplicados, lista de regiones).
        self.undo_stack: list[tuple[np.ndarray, set[int], list[Region]]] = []
        # Los ids nunca se reutilizan, así deshacer no confunde regiones nuevas con antiguas.
        self.next_id = 1
        self.source_path = ''
        self.hojas: list[Hoja] = []
        self.actual: int | None = None
        self._cola: list[int] = []  # hojas que faltan en «Procesar todas»
        self._cola_total = 0
        self._detenida = False
        self._procesando = False  # «Procesar todas» en curso
        self._miniaturas = {}
        self.show_masks = tk.BooleanVar(value=True)
        # Con «Ajustar al texto», un rectángulo arrastrado es solo la zona donde buscar letras.
        self.ajustar = tk.BooleanVar(value=True)
        self.threshold = tk.StringVar(value='Otsu')
        self.method = tk.StringVar(value='Automático')
        self.tool = tk.StringVar(value='Seleccionar')
        self.sensitivity = tk.IntVar(value=4)
        self.expansion = tk.IntVar(value=Options().expansion)
        self.radius = tk.IntVar(value=3)
        self.brush = tk.IntVar(value=5)
        self.zoom = 1.0
        self.panx, self.pany = 0.0, 0.0
        self._preview_img = {}
        self._vistas = {}  # caché de _pagina_de_vista
        self._mapping = {}
        self._drag_start = None
        self._brush_active = False
        self._pan_anchor = None
        self._render_job = None
        self._updating_list = False
        self._advanced_open = False
        self._build()
        root.bind('<Control-o>', lambda _: self.open_image())
        root.bind('<Control-s>', lambda _: self.save_image())
        root.bind('<Control-z>', lambda _: self.undo())
        root.bind('<Delete>', lambda _: self.remove_region())
        root.bind('<Escape>', lambda _: self.change_tool('Seleccionar'))
        root.bind('<Prior>', lambda _: self.ir_a_hoja_relativa(-1))  # Re Pág
        root.bind('<Next>', lambda _: self.ir_a_hoja_relativa(1))  # Av Pág
        root.bind('<Destroy>', self._on_destroy, add='+')
        root.after_idle(self.render_views)

    def _on_destroy(self, event):
        """Cancela un redibujo pendiente cuando se cierra la ventana."""
        if event.widget is self.root and self._render_job is not None:
            self.root.after_cancel(self._render_job)
            self._render_job = None

    def _fit_window(self, width, height):
        """Usa el tamaño pedido si cabe en la pantalla; si no, maximiza la ventana."""
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        if width <= screen_w - 40 and height <= screen_h - 80:
            x = (screen_w - width) // 2
            y = max(0, (screen_h - height) // 2 - 20)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            self.root.minsize(1090, 640)
            return
        # Pantallas chicas (por ejemplo 1366 × 768): ocupar el área útil.
        self.root.geometry(f'{screen_w - 40}x{screen_h - 80}+0+0')
        self.root.minsize(min(1090, screen_w - 40), min(560, screen_h - 80))
        try:
            self.root.state('zoomed')  # Windows
        except tk.TclError:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except tk.TclError:
                pass

    def _new_id(self):
        """Entrega un id interno de región que nunca se ha usado en esta página."""
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def _nuevo_numero(self):
        """Menor número visible (R01, R02…) que no usa ninguna región actual."""
        usados = {r.numero for r in self.regions}
        numero = 1
        while numero in usados:
            numero += 1
        return numero

    @staticmethod
    def _nombre(region):
        """Nombre visible de una región: R01, R02…"""
        return f'R{region.numero:02d}'

    def _push_undo(self, working, applied_ids, regions):
        """Guarda un paso para «Deshacer», limitando la memoria usada en páginas grandes."""
        self.undo_stack.append((working, set(applied_ids), list(regions)))
        max_steps = max(3, min(15, int(400e6 // max(1, working.nbytes))))
        self.undo_stack = self.undo_stack[-max_steps:]

    def _label(self, parent, text, size=10, tone='text', bold=False, **kwargs):
        """Crea una etiqueta con el fondo de su contenedor y un color de la paleta C."""
        return tk.Label(
            parent,
            text=text,
            bg=parent.cget('bg'),
            fg=C.get(tone, tone),
            font=fuente(size, bold),
            **kwargs,
        )

    def _button(self, parent, text, command, kind='normal', help_text=None, **kwargs):
        """Crea un botón redondeado; kind='primary' lo destaca en rosa cerezo."""
        btn = BotonRedondeado(
            parent, text, command, primario=kind == 'primary', **kwargs
        )
        if help_text:
            Tooltip(btn, help_text)
        return btn

    def _header(self):
        """Construye la barra superior: logo, versión y botones Ejemplo, Abrir hojas y Exportar."""
        top = tk.Frame(self.root, bg=C['top'], height=64)
        top.pack(fill='x')
        top.pack_propagate(False)
        logo = tk.Canvas(top, width=38, height=38, bg=C['top'], highlightthickness=0)
        logo.pack(side='left', padx=(20, 11), pady=13)
        self._logo = ImageTk.PhotoImage(
            _rectangulo_redondeado(38, 38, 11, C['accent'], C['top']), master=self.root
        )
        logo.create_image(0, 0, image=self._logo, anchor='nw')
        logo.create_text(19, 19, text='P', fill=C['on_accent'], font=fuente(17, True))
        brand = tk.Frame(top, bg=C['top'])
        brand.pack(side='left')
        self._label(brand, 'PanelClean', 17, 'text', True).pack(anchor='w')
        self._label(brand, f'Editor de globos · {__version__}', 9, 'muted').pack(
            anchor='w'
        )
        self._button(
            top,
            'Exportar',
            self.save_image,
            'primary',
            'Guarda la página abierta, ya limpia, y junto a ella la máscara y un reporte JSON.',
        ).pack(side='right', padx=(0, 18), pady=15)
        self._button(
            top,
            'Abrir hojas',
            self.open_image,
            help_text='Abre una o varias páginas (Ctrl o Shift + clic para elegir varias). '
            'Con varias, quedan en una cola que se puede procesar de una vez.',
        ).pack(side='right', padx=6, pady=15)
        self._button(
            top,
            'Ejemplo',
            self.load_demo,
            help_text='Abre una página de ejemplo para practicar.',
        ).pack(side='right', padx=6, pady=15)

    def _cabecera_panel(self, panel, titulo, flecha, lado):
        """Título de un panel lateral con el botón pequeño que lo oculta."""
        fila = tk.Frame(panel, bg=C['panel'])
        fila.pack(fill='x', padx=16, pady=(14, 6))
        self._label(fila, titulo, 11, 'text', True).pack(side='left')
        self._button(
            fila,
            flecha,
            lambda: self.toggle_panel(lado),
            help_text='Ocultar este panel.',
            alto=24,
            ancho=28,
        ).pack(side='right')
        return fila

    def _group(self, parent, title, pady=0):
        """Agrega un título de sección al panel y devuelve el contenedor de esa sección."""
        frame = tk.Frame(parent, bg=parent.cget('bg'))
        frame.pack(fill='x', pady=(10, pady))
        self._label(frame, title, 10, 'text', True).pack(anchor='w')
        return frame

    def _scale(self, parent, title, var, lo, hi, callback=None):
        """Crea un control deslizante con su valor visible; callback se llama al moverlo."""
        line = tk.Frame(parent, bg=parent.cget('bg'))
        line.pack(fill='x', pady=(8, 0))
        self._label(line, title, 9, 'muted').pack(side='left')
        digit = self._label(line, str(var.get()), 9, 'accent', True)
        digit.pack(side='right')

        def change(val):
            """Actualiza el número mostrado y avisa del cambio."""
            digit.configure(text=str(int(float(val))))
            if callback:
                callback()

        slider = tk.Scale(
            parent,
            from_=lo,
            to=hi,
            orient='horizontal',
            variable=var,
            command=change,
            bg=C['accent'],  # en tk.Scale, bg es el color de la perilla
            fg=C['muted'],
            highlightthickness=0,
            troughcolor=C['panel2'],
            activebackground=C['accent_hover'],
            bd=0,
            showvalue=False,
            sliderlength=18,
            sliderrelief='flat',
            width=10,
        )
        slider.pack(fill='x')
        return slider

    def _combo(self, parent, title, variable, choices, callback):
        """Crea una lista desplegable de solo lectura; callback se llama al elegir una opción."""
        self._label(parent, title, 9, 'muted').pack(anchor='w', pady=(8, 4))
        cb = ttk.Combobox(
            parent, values=choices, textvariable=variable, state='readonly', font=FONT
        )
        cb.pack(fill='x')
        cb.bind('<<ComboboxSelected>>', lambda _: callback())
        return cb

    def _opcion(self, parent, text, variable, value=None, command=None):
        """Botón de opción (value dado) o casilla (sin value) con los colores del panel.

        Al cambiar llama a command (por defecto, redibuja las vistas).
        """
        comun = dict(
            text=text,
            variable=variable,
            command=command or self.render_views,
            bg=parent.cget('bg'),
            fg=C['text'],
            selectcolor=C['bg'],
            activebackground=parent.cget('bg'),
            activeforeground=C['accent'],
            font=SMALL,
            anchor='w',
            highlightthickness=0,
            bd=0,
            relief='flat',
        )
        if value is None:
            return tk.Checkbutton(parent, **comun)
        return tk.Radiobutton(parent, value=value, **comun)

    def _build_left(self, parent):
        """Construye el panel izquierdo: regiones, edición de la máscara y ajustes avanzados.

        El panel no se desplaza: los controles quedan fijos y solo la lista de
        regiones usa el espacio que sobra, con su propia barra si no caben todas.
        """
        panel = tk.Frame(parent, bg=C['panel'], width=268)
        panel.pack_propagate(False)
        self._cabecera_panel(panel, 'Regiones', '‹', 'left')
        self._button(
            panel,
            'Detectar globos',
            self.auto_detect,
            help_text='Búsqueda experimental de globos en toda la página; falla con estilos de dibujo '
            'distintos. Es más seguro encerrar cada globo con el ratón: la selección se ajusta sola al texto.',
        ).pack(fill='x', padx=16, pady=(0, 10))

        # Controles de abajo: se empaquetan antes que la lista para que ella sea
        # la que se achique si falta espacio (por ejemplo, con los ajustes abiertos).
        abajo = tk.Frame(panel, bg=C['panel'])
        abajo.pack(side='bottom', fill='x', padx=16, pady=(0, 14))
        fila = tk.Frame(abajo, bg=C['panel'])
        fila.pack(fill='x', pady=(8, 0))
        self._button(
            fila,
            'Quitar',
            self.remove_region,
            help_text='Quita la región elegida y su limpieza. Se puede deshacer.',
        ).pack(side='left', fill='x', expand=True, padx=(0, 4))
        self._button(
            fila,
            'Quitar todo',
            self.remove_all,
            help_text='Quita todas las regiones y sus limpiezas. Se puede deshacer.',
        ).pack(side='left', fill='x', expand=True, padx=(4, 0))
        self._button(
            abajo,
            'Crear máscaras',
            self.make_selected_mask,
            help_text='Detecta las letras de las regiones elegidas en la lista.',
        ).pack(fill='x', pady=(8, 0))
        self._group(abajo, 'Edición')
        for name, label in [
            ('Seleccionar', 'Seleccionar'),
            ('Pintar', 'Añadir máscara'),
            ('Borrar', 'Borrar máscara'),
        ]:
            self._opcion(abajo, label, self.tool, name).pack(fill='x', pady=(2, 0))
        self._opcion(abajo, 'Ver máscara', self.show_masks).pack(fill='x', pady=(4, 0))
        ajustar = self._opcion(
            abajo, 'Ajustar al texto', self.ajustar, command=self._pista_seleccion
        )
        ajustar.pack(fill='x', pady=(2, 6))
        Tooltip(
            ajustar,
            'Arrastra a grandes rasgos alrededor de uno o varios globos: se crea una región '
            'ajustada a cada bloque de texto. Sin marcar, el rectángulo se usa tal cual.',
        )
        adv_btn = self._button(abajo, 'Ajustes avanzados  ▾', self.toggle_advanced)
        adv_btn.pack(fill='x')
        self.advanced_button = adv_btn
        advanced = tk.Frame(abajo, bg=C['panel'])
        self.advanced_frame = advanced
        self._combo(
            advanced,
            'Umbralización',
            self.threshold,
            ['Otsu', 'Adaptativo'],
            self.on_options_changed,
        )
        self._scale(
            advanced, 'Sensibilidad', self.sensitivity, 1, 8, self.on_options_changed
        )
        self._scale(
            advanced,
            'Expansión de máscara',
            self.expansion,
            0,
            3,
            self.on_options_changed,
        )
        self._scale(advanced, 'Tamaño del pincel', self.brush, 2, 20)

        caja = tk.Frame(panel, bg=C['bg'])
        caja.pack(fill='both', expand=True, padx=16)
        self.reg_list = tk.Listbox(
            caja,
            height=3,
            selectmode=tk.EXTENDED,
            bg=C['bg'],
            fg=C['text'],
            selectbackground=C['select'],
            selectforeground=C['text'],
            bd=0,
            highlightthickness=0,
            relief='flat',
            font=FONT,
            activestyle='none',
            exportselection=False,
        )
        barra = ttk.Scrollbar(caja, orient='vertical', command=self.reg_list.yview)

        def mover_barra(primero, ultimo):
            """Muestra la barra de la lista solo cuando hay más regiones de las que caben."""
            barra.set(primero, ultimo)
            if float(primero) <= 0.0 and float(ultimo) >= 1.0:
                barra.pack_forget()
            elif not barra.winfo_ismapped():
                barra.pack(side='right', fill='y', before=self.reg_list)

        self.reg_list.configure(yscrollcommand=mover_barra)
        self.reg_list.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        self.reg_list.bind('<<ListboxSelect>>', self.on_list_selection)
        Tooltip(self.reg_list, 'Ctrl o Shift + clic para elegir varias regiones.')
        return panel

    def toggle_advanced(self):
        """Muestra u oculta los ajustes avanzados de la máscara."""
        if self._advanced_open:
            self.advanced_frame.pack_forget()
            self.advanced_button.configure(text='Ajustes avanzados  ▾')
        else:
            # Justo debajo del botón; la lista de regiones cede el espacio.
            self.advanced_frame.pack(fill='x', after=self.advanced_button)
            self.advanced_button.configure(text='Ajustes avanzados  ▴')
        self._advanced_open = not self._advanced_open

    def _build_right(self, parent):
        """Construye el panel derecho: datos de la región, método de limpieza, botones y estado."""
        inspector = tk.Frame(parent, bg=C['panel'], width=262)
        inspector.pack_propagate(False)
        self._cabecera_panel(inspector, 'Selección', '›', 'right')
        cuerpo = tk.Frame(inspector, bg=C['panel'])
        cuerpo.pack(fill='both', expand=True, padx=16)
        card = tk.Frame(cuerpo, bg=C['bg'])
        card.pack(fill='x', pady=(2, 0))
        self.info_name = self._label(card, 'Sin región', 12, 'accent', True)
        self.info_name.pack(anchor='w', padx=12, pady=(12, 5))
        self.info_entropy = self._label(card, 'Entropía  —', 9, 'muted')
        self.info_entropy.pack(anchor='w', padx=12, pady=2)
        self.info_dark = self._label(card, 'Tono oscuro  —', 9, 'muted')
        self.info_dark.pack(anchor='w', padx=12, pady=2)
        self.info_pixels = self._label(card, 'Máscara  —', 9, 'muted')
        self.info_pixels.pack(anchor='w', padx=12, pady=(2, 12))
        self._group(cuerpo, 'Limpieza')
        self._combo(
            cuerpo,
            'Método',
            self.method,
            ['Automático', 'Relleno local', 'Inpainting Telea', 'Inpainting NS'],
            self.method_changed,
        )
        self._scale(cuerpo, 'Radio de inpainting', self.radius, 1, 8)
        self._button(
            cuerpo,
            'Limpiar selección',
            self.clean_selected,
            'primary',
            help_text='Limpia las regiones elegidas en la lista y conserva las limpiezas anteriores.',
        ).pack(fill='x', pady=(12, 8))
        self._button(
            cuerpo,
            'Limpiar todas',
            self.clean_all,
            help_text='Intenta limpiar todas las regiones; las dudosas quedan intactas para revisión.',
        ).pack(fill='x', pady=(0, 8))
        self._button(cuerpo, 'Deshacer', self.undo).pack(fill='x')
        self._group(cuerpo, 'Estado')
        self.status_text = self._label(
            cuerpo,
            'Abre una página para comenzar.',
            9,
            'muted',
            wraplength=226,
            justify='left',
        )
        self.status_text.pack(anchor='w', pady=(6, 12))
        return inspector

    def _build_center(self, parent):
        """Construye las vistas ORIGINAL y RESULTADO y conecta el ratón:
        clic izquierdo para seleccionar o pintar, rueda para zoom y clic derecho para desplazar.
        """
        center = tk.Frame(parent, bg=C['bg'])
        nav = tk.Frame(center, bg=C['bg'])
        nav.pack(fill='x', padx=4, pady=(0, 8))
        self._label(nav, 'Editor', 16, 'text', True).pack(side='left')
        self._button(nav, 'Ajustar vista', self.reset_view).pack(side='right')
        self._label(nav, 'Rueda: zoom  ·  clic derecho: mover', 9, 'muted').pack(
            side='right', padx=(0, 13)
        )
        boards = tk.Frame(center, bg=C['bg'])
        boards.pack(fill='both', expand=True)
        self.boards = boards
        self.view_canvases = {}
        for key, title in [('left', 'ORIGINAL'), ('right', 'RESULTADO')]:
            shell = tk.Frame(
                boards,
                bg=C['panel'],
                highlightthickness=1,
                highlightbackground=C['line'],
            )
            shell.pack(
                side='left',
                expand=True,
                fill='both',
                padx=(0, 5) if key == 'left' else (5, 0),
            )
            bar = tk.Frame(shell, bg=C['panel'])
            bar.pack(fill='x', padx=13, pady=(10, 9))
            self._label(bar, title, 9, 'accent', True).pack(side='left')
            if key == 'left':
                self.pista = self._label(bar, '', 9, 'muted')
                self.pista.pack(side='right')
            cv = tk.Canvas(
                shell,
                bg=C['canvas'],
                bd=0,
                highlightthickness=0,
                cursor='crosshair' if key == 'left' else 'fleur',
            )
            cv.pack(fill='both', expand=True, padx=9, pady=(0, 9))
            cv.bind('<Configure>', lambda e: self._schedule_render())
            cv.bind('<ButtonPress-3>', self.pan_start)
            cv.bind('<B3-Motion>', self.pan_move)
            cv.bind('<MouseWheel>', self.mouse_zoom)
            cv.bind('<Button-4>', lambda e: self.adjust_zoom(1.12, e))
            cv.bind('<Button-5>', lambda e: self.adjust_zoom(1 / 1.12, e))
            if key == 'left':
                cv.bind('<ButtonPress-1>', self.left_down)
                cv.bind('<B1-Motion>', self.left_move)
                cv.bind('<ButtonRelease-1>', self.left_up)
            self.view_canvases[key] = cv
        # Botón que aparece en la vista ORIGINAL mientras no hay ninguna página abierta.
        self.add_button = self._button(
            self.view_canvases['left'],
            'Agregar hojas',
            self.open_image,
            'primary',
            alto=42,
            tam=11,
        )
        self.tira_hojas = self._construir_tira(center)
        return center

    def _construir_tira(self, parent):
        """Tira de hojas bajo las vistas: miniaturas de la cola y botones para procesarla.

        Solo se muestra con dos hojas o más (ver _actualizar_tira).
        """
        tira = tk.Frame(parent, bg=C['bg'])
        izquierda = tk.Frame(tira, bg=C['bg'], width=118)
        izquierda.pack(side='left', fill='y', padx=(4, 10))
        izquierda.pack_propagate(False)
        self._label(izquierda, 'Hojas', 11, 'text', True).pack(anchor='w', pady=(8, 0))
        self.contador_hojas = self._label(
            izquierda, '', 9, 'muted', justify='left', wraplength=114
        )
        self.contador_hojas.pack(anchor='w')
        acciones = tk.Frame(tira, bg=C['bg'])
        acciones.pack(side='right', fill='y', padx=(10, 0))
        self.boton_procesar = self._button(
            acciones,
            'Procesar todas',
            self.procesar_todas,
            'primary',
            help_text='Crea las máscaras y limpia en automático todas las hojas, una tras otra. '
            'Las regiones dudosas quedan sin limpiar, para revisarlas.',
            ancho=132,
        )
        self.boton_procesar.pack(pady=(8, 6))
        self._button(
            acciones,
            'Exportar todas',
            self.exportar_todas,
            help_text='Guarda en una carpeta cada hoja limpia, con su máscara y su reporte.',
            ancho=132,
        ).pack()
        cv = tk.Canvas(
            tira,
            height=ALTO_MINIATURA + 16,
            bg=C['panel'],
            bd=0,
            highlightthickness=1,
            highlightbackground=C['line'],
        )
        cv.pack(side='left', fill='x', expand=True, pady=(6, 0))
        cv.configure(xscrollincrement=20)

        def desplazar(paso):
            """Mueve la tira a los lados con la rueda del ratón."""
            cv.xview_scroll(paso, 'units')

        cv.bind('<MouseWheel>', lambda e: desplazar(-2 if e.delta > 0 else 2))
        cv.bind('<Button-4>', lambda e: desplazar(-2))
        cv.bind('<Button-5>', lambda e: desplazar(2))
        self.lienzo_hojas = cv
        return tira

    def _miniatura(self, hoja):
        """Miniatura del resultado de una hoja; se rehace solo si el resultado cambió."""
        guardada = self._miniaturas.get(id(hoja))
        if guardada is not None and guardada[0] is hoja.working:
            return guardada[1]
        alto, ancho = hoja.working.shape[:2]
        tamano = (max(1, round(ancho * ALTO_MINIATURA / alto)), ALTO_MINIATURA)
        chica = cv2.resize(hoja.working, tamano, interpolation=cv2.INTER_AREA)
        foto = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(chica, cv2.COLOR_BGR2RGB)), master=self.root
        )
        self._miniaturas[id(hoja)] = (hoja.working, foto)
        return foto

    def _actualizar_tira(self):
        """Dibuja la tira de hojas: una miniatura por hoja con su número y su estado
        (punto rosa = lista, naranja = revisar, malva = pendiente) y un «+» para sumar
        hojas. Con menos de dos hojas, la tira se oculta.
        """
        if len(self.hojas) < 2:
            if self.tira_hojas.winfo_manager():
                self.tira_hojas.pack_forget()
                self._schedule_render()
            return
        if not self.tira_hojas.winfo_manager():
            self.tira_hojas.pack(
                side='bottom', fill='x', pady=(8, 0), before=self.boards
            )
            self._schedule_render()
        self._guardar_hoja()
        vivas = {id(h) for h in self.hojas}
        self._miniaturas = {k: v for k, v in self._miniaturas.items() if k in vivas}
        cv = self.lienzo_hojas
        cv.delete('all')
        x, arriba, alto = 10, 8, ALTO_MINIATURA
        for i, hoja in enumerate(self.hojas):
            foto = self._miniatura(hoja)
            ancho = foto.width()
            marca = f'hoja{i}'
            es_actual = i == self.actual
            cv.create_rectangle(
                x - 3,
                arriba - 3,
                x + ancho + 2,
                arriba + alto + 2,
                outline=C['accent'] if es_actual else C['line'],
                width=3 if es_actual else 1,
                fill=C['bg'],
                tags=marca,
            )
            cv.create_image(x, arriba, image=foto, anchor='nw', tags=marca)
            # Número de la hoja (abajo a la izquierda) y punto de estado (arriba a la derecha).
            cv.create_rectangle(
                x,
                arriba + alto - 15,
                x + 19,
                arriba + alto,
                fill=C['bg'],
                outline='',
                tags=marca,
            )
            cv.create_text(
                x + 10,
                arriba + alto - 7,
                text=str(i + 1),
                fill=C['text'],
                font=fuente(8, True),
                tags=marca,
            )
            cv.create_oval(
                x + ancho - 13,
                arriba + 3,
                x + ancho - 3,
                arriba + 13,
                fill=C[COLOR_ESTADO[hoja.estado]],
                outline=C['bg'],
                width=2,
                tags=marca,
            )
            cv.tag_bind(marca, '<Button-1>', lambda e, i=i: self.ir_a_hoja(i))
            cv.tag_bind(marca, '<Enter>', lambda e, h=hoja: self._describir_hoja(h))
            cv.tag_bind(marca, '<Leave>', lambda e: self._describir_hoja(None))
            x += ancho + 14
        cv.create_rectangle(
            x - 3,
            arriba - 3,
            x + 38,
            arriba + alto + 2,
            outline=C['accent'],
            dash=(4, 3),
            fill=C['bg'],
            tags='mas',
        )
        cv.create_text(
            x + 17,
            arriba + alto / 2,
            text='+',
            fill=C['accent'],
            font=fuente(18, True),
            tags='mas',
        )
        cv.tag_bind('mas', '<Button-1>', lambda e: self.agregar_hojas())
        cv.tag_bind('mas', '<Enter>', lambda e: self._describir_hoja('mas'))
        cv.tag_bind('mas', '<Leave>', lambda e: self._describir_hoja(None))
        cv.configure(scrollregion=(0, 0, x + 50, alto + 14))
        self._describir_hoja(None)

    def _describir_hoja(self, hoja):
        """Texto bajo «Hojas»: la hoja bajo el cursor o, si no hay, el avance de la cola."""
        if hoja == 'mas':
            texto = 'Agregar más hojas a la cola'
        elif hoja is not None:
            texto = f'{hoja.nombre}\n{hoja.estado}'
        else:
            listas = sum(h.estado == 'Lista' for h in self.hojas)
            revisar = sum(h.estado == 'Revisar' for h in self.hojas)
            texto = f'Hoja {self.actual + 1} de {len(self.hojas)}\n{listas} listas'
            if revisar:
                texto += f', {revisar} por revisar'
        self.contador_hojas.configure(text=texto)

    def _tira(self, parent, flecha, lado):
        """Franja angosta con el botón que vuelve a mostrar un panel oculto."""
        tira = tk.Frame(parent, bg=C['panel'], width=34)
        self._button(
            tira,
            flecha,
            lambda: self.toggle_panel(lado),
            help_text='Mostrar el panel.',
            alto=24,
            ancho=26,
        ).pack(pady=14, padx=4)
        return tira

    def toggle_panel(self, lado):
        """Oculta o vuelve a mostrar un panel lateral ('left' o 'right')."""
        panel, tira = self._paneles[lado]
        columna = 0 if lado == 'left' else 2
        margen = (0, 10) if lado == 'left' else (10, 0)
        if panel.winfo_manager():
            panel.grid_remove()
            tira.grid(row=0, column=columna, sticky='ns', padx=margen)
        else:
            tira.grid_remove()
            panel.grid(row=0, column=columna, sticky='ns', padx=margen)
        self._schedule_render()

    def _build(self):
        """Arma la ventana completa: cabecera, paneles laterales fijos, vistas y pie de página."""
        self._header()
        body = tk.Frame(self.root, bg=C['bg'])
        body.pack(fill='both', expand=True, padx=10, pady=10)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        self.left_panel = self._build_left(body)
        self.center_panel = self._build_center(body)
        self.right_panel = self._build_right(body)
        self.left_panel.grid(row=0, column=0, sticky='ns', padx=(0, 10))
        self.center_panel.grid(row=0, column=1, sticky='nsew')
        self.right_panel.grid(row=0, column=2, sticky='ns', padx=(10, 0))
        self._pista_seleccion()
        self._paneles = {
            'left': (self.left_panel, self._tira(body, '›', 'left')),
            'right': (self.right_panel, self._tira(body, '‹', 'right')),
        }
        footer = tk.Frame(self.root, bg=C['top'], height=30)
        footer.pack(fill='x', side='bottom', before=body)
        footer.pack_propagate(False)
        self.message = self._label(footer, 'Abre una página para comenzar.', 9, 'muted')
        self.message.pack(side='left', padx=16, pady=6)
        self._label(footer, 'PanelClean  ·  TEL-328', 9, 'muted').pack(
            side='right', padx=17
        )

    def _options(self, method=None):
        """Reúne los ajustes actuales de la interfaz; method, si se entrega, reemplaza al método elegido."""
        return Options(
            self.threshold.get(),
            self.sensitivity.get(),
            self.expansion.get(),
            self.radius.get(),
            method or self.method.get(),
        )

    def _set_status(self, text):
        """Muestra el mismo mensaje en el pie de página y en la sección «Estado»."""
        self.message.configure(text=text)
        self.status_text.configure(text=text)

    def open_image(self):
        """Pide una o varias páginas y las abre en lugar de las que hay."""
        rutas = filedialog.askopenfilenames(
            title='Abrir hojas de manga', filetypes=TIPOS_IMAGEN
        )
        if rutas:
            self.abrir_hojas(rutas)

    def agregar_hojas(self):
        """Pide más páginas y las suma al final de la cola."""
        rutas = filedialog.askopenfilenames(
            title='Agregar hojas a la cola', filetypes=TIPOS_IMAGEN
        )
        if rutas:
            self.abrir_hojas(rutas, agregar=True)

    def load_image(self, path):
        """Abre una sola página en lugar de las que hay."""
        return self.abrir_hojas([path])

    def load_demo(self):
        """Abre la página de ejemplo incluida en examples/."""
        path = ROOT / 'examples' / 'demo_pagina.png'
        if path.exists():
            self.load_image(path)

    def abrir_hojas(self, rutas, agregar=False):
        """Abre las páginas dadas y muestra la primera de ellas.

        Reemplazan a las hojas abiertas o, con agregar=True, se suman al final de la
        cola. Antes de reemplazar, avisa si hay limpiezas sin exportar; con una sola
        página de más de 3500 px de lado, pregunta antes de abrirla. Las que no se
        pueden leer se informan y se omiten. Devuelve cuántas hojas se abrieron.
        """
        if self._procesando or (not agregar and not self._descartar_cambios()):
            return 0
        nuevas, fallidas = [], []
        for k, ruta in enumerate(rutas):
            if len(rutas) > 2:
                self._set_status(f'Abriendo hojas: {k + 1} de {len(rutas)}…')
                self.root.update_idletasks()
            try:
                imagen = bgr_from_file(str(ruta))
            except Exception as exc:
                fallidas.append(f'{Path(ruta).name}: {exc}')
                continue
            if (
                len(rutas) == 1
                and max(imagen.shape[:2]) > 3500
                and not messagebox.askyesno(
                    'Imagen grande',
                    'Puede tardar más en procesarse. ¿Abrir de todas formas?',
                )
            ):
                return 0
            nuevas.append(Hoja(str(ruta), imagen, imagen.copy()))
        if fallidas:
            messagebox.showerror(
                (
                    'No se pudo abrir la imagen'
                    if len(rutas) == 1
                    else 'Algunas hojas no se pudieron abrir'
                ),
                '\n'.join(fallidas),
            )
        if not nuevas:
            return 0
        if agregar and self.hojas:
            self._guardar_hoja()
            primera = len(self.hojas)
            self.hojas.extend(nuevas)
        else:
            self.hojas, self.actual, primera = nuevas, None, 0
        self._mostrar_hoja(primera)
        if len(nuevas) > 1:
            self._set_status(
                f'{len(nuevas)} hojas {"agregadas a" if agregar else "en"} la cola. '
                'Encierra el texto de cada una o pulsa «Procesar todas».'
            )
        return len(nuevas)

    def _descartar_cambios(self):
        """Antes de cerrar las hojas abiertas: si alguna tiene limpiezas sin exportar,
        lo avisa. Devuelve True si se puede seguir."""
        self._guardar_hoja()
        sin_exportar = [
            h.nombre for h in self.hojas if h.applied_ids and not h.exportada
        ]
        if not sin_exportar:
            return True
        lista = ', '.join(sin_exportar[:4]) + ('…' if len(sin_exportar) > 4 else '')
        return messagebox.askyesno(
            'Limpiezas sin exportar',
            f'{len(sin_exportar)} hoja(s) tienen limpiezas sin exportar: {lista}.\n\n'
            '¿Cerrarlas de todas formas?',
        )

    def _guardar_hoja(self):
        """Guarda en su Hoja el estado de la hoja abierta (ver CAMPOS_HOJA)."""
        if self.actual is None:
            return
        hoja = self.hojas[self.actual]
        for campo in CAMPOS_HOJA:
            setattr(hoja, campo, getattr(self, campo))

    def _mostrar_hoja(self, indice):
        """Pasa a editar la hoja indice de la cola: guarda la abierta y carga esa."""
        if self.actual is not None and self.actual != indice:
            self._guardar_hoja()
            saliente = self.hojas[self.actual]
            # Una hoja que no se ve conserva solo sus últimos pasos de «Deshacer»,
            # para que una cola larga no llene la memoria.
            saliente.undo_stack = saliente.undo_stack[-PASOS_AL_SALIR:]
        hoja = self.hojas[indice]
        for campo in CAMPOS_HOJA:
            setattr(self, campo, getattr(hoja, campo))
        self.actual = indice
        self.source_path = hoja.ruta
        self._vistas.clear()
        self._drag_start = None
        self._brush_active = False
        self.refresh_regions()
        self.render_views()
        alto, ancho = self.original.shape[:2]
        texto = f'Abierta: {hoja.nombre}  ·  {ancho} × {alto} px'
        if len(self.hojas) > 1:
            texto += f'  ·  hoja {indice + 1} de {len(self.hojas)}'
        self._set_status(texto)

    def ir_a_hoja(self, indice):
        """Abre la hoja indice de la cola (no mientras corre «Procesar todas»)."""
        if (
            self._procesando
            or not 0 <= indice < len(self.hojas)
            or indice == self.actual
        ):
            return
        self._mostrar_hoja(indice)

    def ir_a_hoja_relativa(self, paso):
        """Abre la hoja anterior (paso -1) o la siguiente (paso 1)."""
        if self.actual is not None:
            self.ir_a_hoja(self.actual + paso)

    def _marcar_cambio(self, procesada=False):
        """Anota que el resultado de la hoja abierta cambió: ya no está exportado."""
        if self.actual is None:
            return
        hoja = self.hojas[self.actual]
        hoja.exportada = False
        hoja.procesada = hoja.procesada or procesada

    @staticmethod
    def _rect_overlap(a, b):
        """Fracción del rectángulo más chico que queda cubierta por el otro (de 0 a 1)."""
        x1, y1 = max(a.x, b.x), max(a.y, b.y)
        x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        return area / max(1, min(a.w * a.h, b.w * b.h))

    def auto_detect(self):
        """Agrega como regiones los globos que encuentra detect_bubbles, sin repetir los que ya están."""
        if self.original is None:
            return
        nuevas = self._agregar_encontradas(detect_bubbles(self.original))
        self.refresh_regions()
        self.render_views()
        self._set_status(
            f'{len(nuevas)} posibles globos añadidos. Selecciona las letras dentro de cada globo para mayor precisión.'
        )

    def _agregar_encontradas(self, encontradas, ajustadas=False):
        """Suma a la hoja las regiones encontradas que no repiten una existente, y las
        deja elegidas. Devuelve las que se sumaron.

        Un globo detectado se descarta si cae sobre una región que ya está. Un bloque
        ajustado al texto (ajustadas=True) solo si es casi el mismo rectángulo que otra
        región: dentro de un globo grande ya marcado sí se agrega, porque es más preciso.
        """
        nuevas = []
        for candidata in encontradas:
            if any(
                (
                    iou_rect(candidata, r) > 0.5
                    if ajustadas
                    else self._rect_overlap(candidata, r) > 0.5
                )
                for r in self.regions
            ):
                continue
            candidata.id = self._new_id()
            candidata.numero = self._nuevo_numero()
            self.regions.append(candidata)
            self.selected_indices.add(len(self.regions) - 1)
            nuevas.append(candidata)
        if nuevas:
            self.focus_index = len(self.regions) - 1
        return nuevas

    def selected_regions(self):
        """Devuelve las regiones seleccionadas en la lista, en orden."""
        return [
            self.regions[i]
            for i in sorted(self.selected_indices)
            if 0 <= i < len(self.regions)
        ]

    def current_region(self):
        """Devuelve la región con foco (la que describe el panel derecho), o None."""
        if self.focus_index is not None and 0 <= self.focus_index < len(self.regions):
            return self.regions[self.focus_index]
        return None

    def refresh_regions(self):
        """Redibuja la lista de regiones y su selección. ✓ = limpieza aplicada, • = tiene máscara."""
        self._updating_list = True
        self.reg_list.delete(0, 'end')
        for reg in self.regions:
            state = (
                '✓'
                if reg.id in self.applied_ids
                else ('•' if reg.mask is not None else ' ')
            )
            self.reg_list.insert(
                'end', f' {state}  {self._nombre(reg)}   {reg.w}×{reg.h}'
            )
        self.selected_indices = {
            i for i in self.selected_indices if i < len(self.regions)
        }
        self.reg_list.selection_clear(0, 'end')
        for index in sorted(self.selected_indices):
            self.reg_list.selection_set(index)
        if self.focus_index is not None and self.focus_index < len(self.regions):
            self.reg_list.activate(self.focus_index)
            self.reg_list.see(self.focus_index)
        self._updating_list = False
        self.refresh_inspector()
        self._actualizar_tira()

    def on_list_selection(self, event=None):
        """Sincroniza la selección cuando el usuario hace clic en la lista de regiones."""
        if self._updating_list:
            return
        self.selected_indices = set(self.reg_list.curselection())
        if self.selected_indices:
            active = self.reg_list.index(tk.ACTIVE)
            self.focus_index = (
                active
                if active in self.selected_indices
                else max(self.selected_indices)
            )
            reg = self.current_region()
            if reg:
                self.method.set(reg.method_override)
        else:
            self.focus_index = None
        self.refresh_inspector()
        self.render_views()

    def refresh_inspector(self):
        """Actualiza el panel «Selección» con la clase de fondo y las medidas de la región con foco."""
        r = self.current_region()
        if r is None:
            self.info_name.configure(text='Sin región', fg=C['muted'])
            self.info_entropy.configure(text='Entropía  —')
            self.info_dark.configure(text='Tono oscuro  —')
            self.info_pixels.configure(text='Máscara  —')
            return
        if r.mask is not None:
            describe_background(self.original, r)
        self.info_name.configure(
            text=f'{self._nombre(r)}  ·  {r.classification}',
            fg=C['accent'] if not r.review_reason else C['warn'],
        )
        self.info_entropy.configure(
            text=f'Entropía  {r.entropy:.1f} bits  ·  textura {r.texture:.0%}'
        )
        self.info_dark.configure(text=f'Tono oscuro  {r.dark_level:.0f} / 255')
        self.info_pixels.configure(
            text=f'Máscara  {cv2.countNonZero(r.mask) if r.mask is not None else 0:,} px'
        )

    def on_options_changed(self):
        """Avisa que hay que recalcular las máscaras tras cambiar un ajuste."""
        self._set_status(
            'Ajustes cambiados. Usa «Crear máscaras» para recalcular la selección.'
        )

    def method_changed(self):
        """Asigna el método elegido en la lista a todas las regiones seleccionadas."""
        regs = self.selected_regions()
        for r in regs:
            r.method_override = self.method.get()
        if regs:
            self._set_status(
                f'Método «{self.method.get()}» asignado a {len(regs)} región(es).'
            )

    def make_selected_mask(self):
        """Calcula la máscara de las regiones seleccionadas.

        Si alguna tiene correcciones hechas con el pincel, pregunta antes de perderlas.
        """
        regs = self.selected_regions()
        if self.original is None or not regs:
            messagebox.showinfo(
                'Seleccionar letras', 'Marca una o varias regiones alrededor del texto.'
            )
            return
        edited = [r for r in regs if r.mask_edited]
        if edited and not messagebox.askyesno(
            'Correcciones manuales',
            f'{len(edited)} región(es) tienen correcciones hechas con el pincel.\n\n'
            '¿Recalcular sus máscaras y perder esas correcciones?',
        ):
            regs = [r for r in regs if not r.mask_edited]
            if not regs:
                self._set_status('Se conservaron las máscaras corregidas a mano.')
                return
        counts = []
        for r in regs:
            mask = generate_mask(self.original, r, self._options())
            counts.append(cv2.countNonZero(mask))
        self.refresh_regions()
        self.render_views()
        self._set_status(
            f'Máscaras creadas para {len(regs)} regiones ({sum(counts):,} px). Verifica la superposición rosa antes de limpiar.'
        )

    def remove_region(self):
        """Quita la región elegida (la que describe el panel derecho) y su limpieza. Se puede deshacer."""
        region = self.current_region()
        if region is None:
            return
        self._quitar([region])
        self._set_status(
            f'{self._nombre(region)} quitada. Se actualizó el resultado; puedes deshacerlo.'
        )

    def remove_all(self):
        """Quita todas las regiones y sus limpiezas. Se puede deshacer."""
        if not self.regions:
            return
        cantidad = len(self.regions)
        self._quitar(list(self.regions))
        self._set_status(
            f'Se quitaron las {cantidad} regiones. Se actualizó el resultado; puedes deshacerlo.'
        )

    def _quitar(self, regiones):
        """Quita las regiones dadas, guarda un paso de «Deshacer» y recompone el resultado."""
        self._push_undo(self.working.copy(), self.applied_ids, self.regions)
        for r in regiones:
            self.applied_ids.discard(r.id)
        self.regions = [r for r in self.regions if all(r is not q for q in regiones)]
        self.selected_indices = set()
        self.focus_index = None
        self._rebuild_without_snapshot()
        self._marcar_cambio()
        self.refresh_regions()
        self.render_views()

    def change_tool(self, name):
        """Cambia la herramienta activa: Seleccionar, Pintar o Borrar."""
        self.tool.set(name)
        self._drag_start = None
        self.render_views()

    def _to_image_xy(self, event):
        """Convierte un clic en la vista ORIGINAL a coordenadas de la imagen (None si no hay imagen)."""
        if self.original is None:
            return None
        s, ox, oy = self._mapping.get('left', (0, 0, 0))
        if s <= 0:
            return None
        x = int((event.x - ox) / s)
        y = int((event.y - oy) / s)
        return max(0, min(self.original.shape[1] - 1, x)), max(
            0, min(self.original.shape[0] - 1, y)
        )

    def left_down(self, event):
        """Inicio de un clic: empieza un rectángulo o un trazo de pincel, según la herramienta."""
        point = self._to_image_xy(event)
        if point is None:
            return
        if self.tool.get() == 'Seleccionar':
            self._drag_start = point
        else:
            self._brush_active = True
            self.paint_mask(point)

    def left_move(self, event):
        """Arrastre: dibuja el rectángulo provisional o sigue pintando la máscara."""
        point = self._to_image_xy(event)
        if point is None:
            return
        if self.tool.get() == 'Seleccionar' and self._drag_start:
            scale, ox, oy = self._mapping.get('left', (1, 0, 0))
            x, y = self._drag_start
            cv = self.view_canvases['left']
            cv.delete('rubber')
            cv.create_rectangle(
                ox + x * scale,
                oy + y * scale,
                ox + point[0] * scale,
                oy + point[1] * scale,
                outline=C['accent'],
                width=2,
                dash=(5, 3),
                tags='rubber',
            )
        elif self._brush_active:
            self.paint_mask(point)

    def left_up(self, event):
        """Fin del clic. Un clic corto selecciona la región bajo el cursor (con Ctrl la suma
        o la quita de la selección); un arrastre crea una región nueva. Con el pincel,
        termina el trazo.
        """
        point = self._to_image_xy(event)
        if self.tool.get() != 'Seleccionar' or self._drag_start is None:
            if self._brush_active:
                self.end_brush()
            return
        start = self._drag_start
        self._drag_start = None
        self.view_canvases['left'].delete('rubber')
        if point is None or self.original is None:
            return
        if abs(start[0] - point[0]) < 9 and abs(start[1] - point[1]) < 9:
            for i in range(len(self.regions) - 1, -1, -1):
                r = self.regions[i]
                if r.contains(*point):
                    if event.state & 0x0004:
                        if i in self.selected_indices:
                            self.selected_indices.remove(i)
                        else:
                            self.selected_indices.add(i)
                    else:
                        self.selected_indices = {i}
                    self.focus_index = (
                        i
                        if i in self.selected_indices
                        else (
                            max(self.selected_indices)
                            if self.selected_indices
                            else None
                        )
                    )
                    rr = self.current_region()
                    if rr:
                        self.method.set(rr.method_override)
                    self.refresh_regions()
                    self.render_views()
                    return
        else:
            try:
                self.add_zone((start[0], start[1], point[0], point[1]))
            except ValueError as exc:
                self._set_status(str(exc))

    def _pista_seleccion(self):
        """Indica sobre la vista ORIGINAL cómo marcar el texto, según «Ajustar al texto»."""
        self.pista.configure(
            text=(
                'Encierra el globo: se ajusta al texto'
                if self.ajustar.get()
                else 'Selecciona texto, no el contorno'
            )
        )

    def add_zone(self, xyxy):
        """Crea regiones a partir del rectángulo (x1, y1, x2, y2) que arrastró el usuario.

        Con «Ajustar al texto», el rectángulo es solo la zona donde buscar: se crea una
        región ajustada por cada bloque de texto en globos que haya adentro
        (texto_en_zona), sin repetir las que ya existen. Si no encuentra ninguno, o la
        casilla está desmarcada, usa el rectángulo tal cual. Devuelve las regiones nuevas.
        """
        if not self.ajustar.get():
            return [self.add_region(xyxy)]
        bloques = texto_en_zona(self.original, xyxy)
        if not bloques:
            region = self.add_region(xyxy)
            self._set_status(
                f'No se encontró texto en globos dentro de la zona; {self._nombre(region)} '
                'usa el rectángulo tal como lo marcaste.'
            )
            return [region]
        nuevas = self._agregar_encontradas(bloques, ajustadas=True)
        if nuevas:
            self.method.set('Automático')
        self.refresh_regions()
        self.render_views()
        nombres = ', '.join(self._nombre(r) for r in nuevas)
        self._set_status(
            f'Texto encontrado en la zona: {nombres}. Revisa los rectángulos y crea las máscaras.'
            if nuevas
            else 'El texto de esa zona ya tiene regiones.'
        )
        return nuevas

    def add_region(self, xyxy):
        """Crea una región con el rectángulo (x1, y1, x2, y2), la selecciona y la devuelve.

        Lanza ValueError si el rectángulo queda menor a 12 × 12 píxeles.
        """
        r = bounded_region(self.original, xyxy)
        r.id = self._new_id()
        r.numero = self._nuevo_numero()
        self.regions.append(r)
        index = len(self.regions) - 1
        self.selected_indices.add(index)
        self.focus_index = index
        self.method.set('Automático')
        self.refresh_regions()
        self.render_views()
        self._set_status(
            f'{self._nombre(r)} añadida. Se mantienen {len(self.selected_indices)} selecciones; crea las máscaras.'
        )
        return r

    def paint_mask(self, point):
        """Pinta o borra un círculo de la máscara de la región con foco, en el punto dado."""
        if self.original is None:
            return
        r = self.current_region()
        if r is None or not r.contains(*point):
            return
        if r.mask is None:
            generate_mask(self.original, r, self._options())
        x, y = point[0] - r.x, point[1] - r.y
        cv2.circle(
            r.mask,
            (x, y),
            self.brush.get(),
            255 if self.tool.get() == 'Pintar' else 0,
            -1,
        )
        r.mask = cv2.bitwise_and(r.mask, inner_area(r))
        r.mask_edited = True
        # Durante el trazo solo se redibuja (agrupando eventos); el fondo se
        # vuelve a analizar una sola vez, al soltar el botón (end_brush).
        self._schedule_render(30)

    def end_brush(self):
        """Termina el trazo de pincel: vuelve a analizar el fondo y actualiza las vistas."""
        self._brush_active = False
        self.refresh_inspector()
        self.render_views()

    def clean_selected(self):
        """Limpia las regiones seleccionadas en la lista."""
        regs = self.selected_regions()
        if not regs:
            messagebox.showinfo(
                'Sin selección',
                'Selecciona una o varias regiones de texto en la lista.',
            )
            return
        self._perform_clean(regs)

    def clean_all(self):
        """Intenta limpiar todas las regiones; las dudosas quedan sin aplicar."""
        if not self.regions:
            messagebox.showinfo(
                'Sin regiones', 'Selecciona el texto de uno o más globos.'
            )
            return
        self._perform_clean(self.regions)

    def _rebuild_without_snapshot(self):
        """Recompone el resultado desde la imagen original aplicando, en orden, las regiones de applied_ids.

        No guarda un paso de «Deshacer» (eso lo hace quien la llama). Si una región ya
        no se puede limpiar (por ejemplo, su máscara quedó vacía), se marca como no aplicada.
        """
        if self.original is None:
            return
        self.working = self.original.copy()
        for r in self.regions:
            if r.id not in self.applied_ids:
                continue
            opts = self._options(r.method_override)
            # Cada región conserva el radio con que se limpió, aunque luego cambie el control.
            if r.applied_radius is not None:
                opts.radius = r.applied_radius
            result, method, n = clean_region(self.working, r, opts)
            if n > 0:
                self.working = result
                r.last_method = method
            else:
                self.applied_ids.discard(r.id)

    def _perform_clean(self, regions, preguntar=True):
        """Limpia las regiones dadas y guarda un paso de «Deshacer».

        Las regiones que quedan en «Revisión manual» o sin máscara no se aplican. El
        resultado se recompone siempre desde el original, para no aplicar inpainting
        sobre zonas ya limpiadas. Con preguntar=False (la cola de hojas), las regiones
        para revisar con un método elegido a mano se dejan sin limpiar en vez de
        pedir confirmación.
        """
        if self.original is None:
            return
        try:
            # 1) Toda región necesita máscara y un análisis del fondo actualizado,
            #    porque de ese análisis sale el método «Automático».
            for r in regions:
                if r.mask is None:
                    generate_mask(self.original, r, self._options())
                else:
                    describe_background(self.original, r)
            # Estado previo, para poder deshacer esta limpieza.
            before = self.working.copy()
            before_ids = set(self.applied_ids)
            before_regions = list(self.regions)
            methods = []
            applied = 0
            skipped = []
            # 2) Decidir qué regiones se aplican. Luego se recompone todo desde el
            #    original: así no se re-aplica inpainting y las limpiezas
            #    anteriores no se degradan.
            for r in regions:
                opts = self._options(r.method_override)
                chosen = (
                    r.suggested_method if opts.method == 'Automático' else opts.method
                )
                if (
                    chosen == 'Revisión manual'
                    or r.mask is None
                    or not cv2.countNonZero(r.mask)
                ):
                    self.applied_ids.discard(r.id)
                    skipped.append(self._nombre(r))
                else:
                    if (
                        opts.method != 'Automático'
                        and r.classification == 'Revisar máscara'
                    ):
                        confirmed = preguntar and messagebox.askyesno(
                            'Confirmar limpieza',
                            f'{self._nombre(r)}: {r.review_reason}\n\n'
                            '¿Deseas aplicar el método elegido de todas formas?',
                        )
                        if not confirmed:
                            self.applied_ids.discard(r.id)
                            skipped.append(self._nombre(r))
                            continue
                    self.applied_ids.add(r.id)
                    r.applied_radius = opts.radius
            self._rebuild_without_snapshot()
            self._marcar_cambio(procesada=True)
            # 3) Resumen para la barra de estado.
            for r in regions:
                if r.id in self.applied_ids:
                    applied += 1
                    methods.append(r.last_method)
                elif self._nombre(r) not in skipped:
                    skipped.append(self._nombre(r))
            if (
                not np.array_equal(before, self.working)
                or self.applied_ids != before_ids
            ):
                self._push_undo(before, before_ids, before_regions)
            self.refresh_regions()
            self.render_views()
            msg = f'{applied} región(es) limpiada(s).'
            if skipped:
                msg += f' Pendientes de revisión: {", ".join(skipped)}.'
            if applied:
                msg += f' {", ".join(dict.fromkeys(methods))}.'
            self._set_status(msg)
        except Exception as exc:
            messagebox.showerror('No se pudo completar la limpieza', str(exc))

    def undo(self):
        """Vuelve al paso anterior: imagen resultado, regiones aplicadas y lista de regiones."""
        if not self.undo_stack:
            self._set_status('No hay cambios recientes para deshacer.')
            return
        working, applied, regions = self.undo_stack.pop()
        # Si el paso deshecho cambió la lista de regiones, la selección ya no es válida.
        if [id(r) for r in regions] != [id(r) for r in self.regions]:
            self.selected_indices = set()
            self.focus_index = None
        self.working, self.applied_ids, self.regions = working, applied, regions
        self._marcar_cambio()
        self.refresh_regions()
        self.render_views()
        self._set_status('Limpieza deshecha. Las regiones y máscaras siguen editables.')

    def procesar_todas(self):
        """Limpia en automático todas las hojas de la cola, una tras otra.

        Cada hoja usa sus regiones; en las que no tienen ninguna, si el usuario lo
        acepta, se buscan globos (detect_bubbles, experimental). Se crean las máscaras
        que falten y cada región se limpia con su método (Automático por defecto):
        las dudosas quedan sin limpiar, para revisarlas. Se procesa una hoja por vez
        con after(), así la ventana responde y se ve el avance; pulsar otra vez el
        botón («Detener») vacía la cola.
        """
        if self._procesando:
            self._cola = []
            self._detenida = True
            return
        if not self.hojas:
            return
        self._guardar_hoja()
        sin_regiones = sum(not h.regions for h in self.hojas)
        buscar = True
        if sin_regiones:
            buscar = messagebox.askyesnocancel(
                'Hojas sin regiones',
                f'{sin_regiones} de {len(self.hojas)} hojas no tienen regiones marcadas.\n\n'
                '¿Buscar globos en ellas automáticamente? La búsqueda es experimental: '
                'revisa esas hojas al terminar.\n\n'
                'Sí: buscar y limpiar.   No: procesar solo las hojas con regiones.',
            )
            if buscar is None:
                return
        self._cola = [i for i, h in enumerate(self.hojas) if h.regions or buscar]
        self._cola_total = len(self._cola)
        self._detenida = False
        if self._cola:
            self._procesando = True
            self.boton_procesar.configure(text='Detener')
            self.root.after(1, self._procesar_siguiente)

    def _procesar_siguiente(self):
        """Procesa la siguiente hoja de «Procesar todas» y programa la que sigue."""
        if not self._cola:
            self._terminar_cola()
            return
        indice = self._cola.pop(0)
        self._mostrar_hoja(indice)
        hechas = self._cola_total - len(self._cola)
        self._set_status(f'Procesando hojas: {hechas} de {self._cola_total}…')
        self.root.update_idletasks()
        if not self.regions:
            self._agregar_encontradas(detect_bubbles(self.original))
        if self.regions:
            self._perform_clean(list(self.regions), preguntar=False)
        else:
            self._marcar_cambio(
                procesada=True
            )  # no se encontró texto: queda por revisar
            self.refresh_regions()
        self.root.after(15, self._procesar_siguiente)

    def _terminar_cola(self):
        """Cierra «Procesar todas»: restaura el botón y resume cómo quedó cada hoja."""
        self._procesando = False
        self.boton_procesar.configure(text='Procesar todas')
        self._guardar_hoja()
        listas = sum(h.estado == 'Lista' for h in self.hojas)
        revisar = [h.nombre for h in self.hojas if h.estado == 'Revisar']
        texto = 'Cola detenida' if self._detenida else 'Cola procesada'
        texto += f': {listas} hoja(s) listas'
        if revisar:
            texto += f', {len(revisar)} por revisar (punto naranja)'
        self._set_status(texto + '. Revisa las hojas y luego pulsa «Exportar todas».')
        self._actualizar_tira()

    def exportar_todas(self):
        """Guarda en una carpeta cada hoja con alguna limpieza aplicada, como
        <nombre>_limpia.png con su máscara y su reporte (ver _escribir_resultado)."""
        if self._procesando or not self.hojas:
            return
        self._guardar_hoja()
        con_limpieza = [h for h in self.hojas if h.applied_ids]
        if not con_limpieza:
            messagebox.showinfo(
                'Nada que exportar', 'Ninguna hoja tiene limpiezas aplicadas todavía.'
            )
            return
        carpeta = filedialog.askdirectory(
            title='Carpeta donde guardar las hojas limpias', mustexist=True
        )
        if not carpeta:
            return
        destinos, usados = [], set()
        for hoja in con_limpieza:
            # Dos hojas con el mismo nombre (de carpetas distintas) no se pisan.
            base = nombre = f'{Path(hoja.ruta).stem}_limpia'
            numero = 2
            while nombre.lower() in usados:
                nombre = f'{base}_{numero}'
                numero += 1
            usados.add(nombre.lower())
            destinos.append((hoja, Path(carpeta) / f'{nombre}.png'))
        existentes = [d.name for _, d in destinos if d.exists()]
        if existentes and not messagebox.askyesno(
            'Reemplazar archivos',
            f'{len(existentes)} archivo(s) ya existen en esa carpeta, por ejemplo '
            f'{existentes[0]}.\n\n¿Reemplazarlos?',
        ):
            return
        try:
            for k, (hoja, destino) in enumerate(destinos):
                self._set_status(f'Exportando hojas: {k + 1} de {len(destinos)}…')
                self.root.update_idletasks()
                self._escribir_resultado(destino, hoja)
                hoja.exportada = True
        except Exception as exc:
            messagebox.showerror('Error al exportar', str(exc))
            return
        detalle = f'Se guardaron {len(destinos)} hojas en:\n{carpeta}'
        revisar = [h.nombre for h, _ in destinos if h.estado == 'Revisar']
        sin_limpiar = [h.nombre for h in self.hojas if not h.applied_ids]
        if revisar:
            detalle += f'\n\nCon regiones sin limpiar: {", ".join(revisar)}.'
        if sin_limpiar:
            detalle += f'\n\nSin limpiezas, no se exportaron: {", ".join(sin_limpiar)}.'
        self._set_status(f'Se exportaron {len(destinos)} hojas.')
        messagebox.showinfo('Exportación completada', detalle)

    def save_image(self):
        """Exporta la hoja abierta: imagen limpia (PNG), máscara (_mascara.png) y reporte (_reporte.json)."""
        if self.working is None:
            return
        self._guardar_hoja()
        hoja = self.hojas[self.actual]
        dest = filedialog.asksaveasfilename(
            defaultextension='.png',
            initialfile=f'{Path(hoja.ruta).stem}_limpia.png',
            filetypes=[('Imagen PNG', '*.png')],
            title='Exportar página limpia',
        )
        if not dest:
            return
        try:
            nombres = self._escribir_resultado(dest, hoja)
            hoja.exportada = True
            self._set_status('Se exportaron la imagen, máscara y reporte.')
            messagebox.showinfo(
                'Exportación completada', 'Guardados:\n' + '\n'.join(nombres)
            )
        except Exception as exc:
            messagebox.showerror('Error al exportar', str(exc))

    def _escribir_resultado(self, destino, hoja):
        """Escribe la hoja limpia en destino (PNG) y, junto a ella, la máscara de las
        regiones aplicadas (<nombre>_mascara.png) y un reporte (<nombre>_reporte.json).
        Devuelve los nombres de los tres archivos."""
        destino = Path(destino)
        stem = str(destino.with_suffix(''))
        write_png(str(destino), hoja.working)
        aplicadas = [r for r in hoja.regions if r.id in hoja.applied_ids]
        mask = mask_whole(hoja.original, aplicadas)
        write_png(stem + '_mascara.png', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        info = {
            'proyecto': 'PanelClean',
            'version': __version__,
            'origen': hoja.nombre,
            'regiones': [
                {
                    'region': self._nombre(r),
                    'rect': r.rect,
                    'procesada': r.id in hoja.applied_ids,
                    'clase': r.classification,
                    'metodo': r.last_method,
                    'entropia': round(r.entropy, 3),
                    'tono_oscuro': round(r.dark_level, 1),
                    'textura': round(r.texture, 3),
                    'varianza': round(r.variance, 1),
                    'densidad_bordes': round(r.edge_density, 4),
                    'pixeles_mascara': (
                        int(cv2.countNonZero(r.mask)) if r.mask is not None else 0
                    ),
                }
                for r in hoja.regions
            ],
        }
        Path(stem + '_reporte.json').write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf8'
        )
        return [
            destino.name,
            Path(stem + '_mascara.png').name,
            Path(stem + '_reporte.json').name,
        ]

    def reset_view(self):
        """Vuelve al zoom inicial y centra las vistas."""
        self.panx = self.pany = 0.0
        self.zoom = 1.0
        self.render_views()

    def adjust_zoom(self, factor, event=None):
        """Multiplica el zoom por factor, manteniéndolo entre 0,65× y 8×.

        Si se entrega el evento del ratón, el punto de la página bajo el cursor
        queda fijo (se acerca hacia donde apunta el ratón); si no, hacia el centro.
        """
        nuevo = max(0.65, min(8.0, self.zoom * factor))
        if event is not None and self.original is not None:
            key = 'left' if event.widget is self.view_canvases['left'] else 'right'
            scale, ox, oy = self._mapping.get(key, (0, 0, 0))
            if scale > 0:
                # Punto de la página bajo el cursor, antes del zoom.
                px, py = (event.x - ox) / scale, (event.y - oy) / scale
                canvas = event.widget
                w, h = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
                ih, iw = self.original.shape[:2]
                nueva = scale * nuevo / self.zoom
                # Desplazamiento que deja ese punto otra vez bajo el cursor.
                self.panx = event.x - px * nueva - (w - iw * nueva) / 2
                self.pany = event.y - py * nueva - (h - ih * nueva) / 2
        self.zoom = nuevo
        self.render_views()

    def mouse_zoom(self, event):
        """Acerca o aleja con la rueda del ratón, hacia el punto bajo el cursor
        (Windows y macOS; Linux usa los botones 4 y 5)."""
        self.adjust_zoom(1.13 if event.delta > 0 else 1 / 1.13, event)

    def pan_start(self, event):
        """Guarda el punto donde empieza el desplazamiento con el botón derecho."""
        self._pan_anchor = (event.x, event.y)

    def pan_move(self, event):
        """Desplaza ambas vistas siguiendo el ratón.

        Lo ya dibujado se mueve al instante; el redibujo (que completa los bordes que
        quedan a la vista) se agrupa para no hacerlo en cada evento del ratón.
        """
        if self._pan_anchor is None or self.original is None:
            return
        ax, ay = self._pan_anchor
        dx, dy = event.x - ax, event.y - ay
        self.panx += dx
        self.pany += dy
        self._pan_anchor = (event.x, event.y)
        for key, canvas in self.view_canvases.items():
            canvas.move('all', dx, dy)
            if key in self._mapping:
                scale, ox, oy = self._mapping[key]
                self._mapping[key] = (scale, ox + dx, oy + dy)
        self._schedule_render(15)

    def _schedule_render(self, delay=120):
        """Programa un redibujo dentro de delay milisegundos, si no hay uno pendiente."""
        # Agrupa muchos eventos seguidos (redimensionar, pincel, mover) en un solo dibujo.
        if self._render_job is not None:
            return
        self._render_job = self.root.after(delay, self._run_scheduled_render)

    def _run_scheduled_render(self):
        """Ejecuta el redibujo programado."""
        self._render_job = None
        self.render_views()

    def _pagina_de_vista(self, key):
        """Página que muestra una vista, en RGB y a tamaño real; la izquierda lleva
        encima las máscaras en rosa.

        Se guarda en caché y solo se recalcula cuando cambia lo que muestra (otra
        página, otro resultado u otras máscaras), no al acercar ni desplazar la vista.
        La firma usa los id() de los arreglos: la caché los mantiene vivos, así un id
        no se puede repetir mientras siga guardado, y ni el resultado ni las máscaras
        se modifican en el lugar (siempre se reemplazan por un arreglo nuevo).
        """
        if key == 'right':
            fuentes = (self.working,)
            firma = (id(self.working),)
        else:
            con_mascara = (
                [r for r in self.regions if r.mask is not None]
                if self.show_masks.get()
                else []
            )
            fuentes = (self.original, *[r.mask for r in con_mascara])
            firma = (id(self.original),) + tuple(
                (id(r.mask), r.x, r.y) for r in con_mascara
            )
        guardada = self._vistas.get(key)
        if guardada is not None and guardada[0] == firma:
            return guardada[2]
        pagina = cv2.cvtColor(fuentes[0], cv2.COLOR_BGR2RGB)
        if key == 'left':
            for r in con_mascara:
                zona = pagina[r.y : r.y + r.h, r.x : r.x + r.w]
                marcada = r.mask > 0
                zona[marcada] = (
                    zona[marcada] * (1 - ALFA_MASCARA) + ROSA_MASCARA * ALFA_MASCARA
                ).astype(np.uint8)
        self._vistas[key] = (firma, fuentes, pagina, {})
        return pagina

    def _copia_reducida(self, key, pagina, scale):
        """Copia de la página ya achicada a la mitad, un cuarto, etc., para que al ver
        la hoja entera no haya que promediar millones de píxeles en cada redibujo.

        Devuelve la copia y la escala que lleva de ella a la pantalla. Las copias se
        guardan junto a la página en la caché de _pagina_de_vista.
        """
        factor = 1
        while scale * factor * 2 <= 1:
            factor *= 2
        if factor == 1:
            return pagina, scale
        copias = self._vistas[key][3]
        if factor not in copias:
            ih, iw = pagina.shape[:2]
            copias[factor] = cv2.resize(
                pagina,
                (max(1, iw // factor), max(1, ih // factor)),
                interpolation=cv2.INTER_AREA,
            )
        copia = copias[factor]
        return copia, scale * pagina.shape[1] / copia.shape[1]

    def render_views(self):
        """Dibuja la página original (con máscaras y rectángulos) y el resultado.

        Solo se escala la parte de la página que cabe en la vista: con zoom, escalar la
        página entera en cada movimiento era lo que hacía lento el desplazamiento.
        Guarda en _mapping la escala y el desplazamiento de cada vista, que
        _to_image_xy usa para convertir clics a coordenadas de la imagen.
        Sin ninguna página abierta, muestra el botón «Agregar hojas».
        """
        if self.original is None:
            self._render_sin_imagen()
            return
        for key in ('left', 'right'):
            pagina = self._pagina_de_vista(key)
            canvas = self.view_canvases[key]
            w, h = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
            ih, iw = pagina.shape[:2]
            # Escala que hace caber la página en la vista (con 11 px de margen por
            # lado), multiplicada por el zoom; luego se centra y se suma el desplazamiento.
            scale = min((w - 22) / iw, (h - 22) / ih) * self.zoom
            ox = (w - iw * scale) / 2 + self.panx
            oy = (h - ih * scale) / 2 + self.pany
            self._mapping[key] = (scale, ox, oy)
            canvas.delete('all')
            fuente_px, escala_px = self._copia_reducida(key, pagina, scale)
            sh, sw = fuente_px.shape[:2]
            # Parte visible de la página, en píxeles de la copia que se escala.
            x0, y0 = max(0, int(-ox / escala_px)), max(0, int(-oy / escala_px))
            x1 = min(sw, int(math.ceil((w - ox) / escala_px)))
            y1 = min(sh, int(math.ceil((h - oy) / escala_px)))
            if x1 > x0 and y1 > y0:
                tamano = (
                    max(1, round((x1 - x0) * escala_px)),
                    max(1, round((y1 - y0) * escala_px)),
                )
                visible = cv2.resize(
                    fuente_px[y0:y1, x0:x1],
                    tamano,
                    # INTER_AREA promedia al achicar (sin dientes de sierra).
                    interpolation=(
                        cv2.INTER_AREA if escala_px < 1 else cv2.INTER_CUBIC
                    ),
                )
                self._preview_img[key] = ImageTk.PhotoImage(
                    Image.fromarray(visible), master=self.root
                )
                canvas.create_image(
                    round(ox + x0 * escala_px),
                    round(oy + y0 * escala_px),
                    image=self._preview_img[key],
                    anchor='nw',
                )
            if key == 'left':
                for i, r in enumerate(self.regions):
                    selected = i in self.selected_indices
                    color = C['accent'] if selected else C['region']
                    lx, ty = ox + r.x * scale, oy + r.y * scale
                    rx, by = lx + r.w * scale, ty + r.h * scale
                    canvas.create_rectangle(
                        lx, ty, rx, by, outline=color, width=2.3 if selected else 1.4
                    )
                    canvas.create_rectangle(
                        lx, ty - 19, lx + 39, ty, fill=color, outline=color
                    )
                    canvas.create_text(
                        lx + 19,
                        ty - 10,
                        text=self._nombre(r),
                        fill=C['on_accent'],
                        font=fuente(9, True),
                    )

    def _render_sin_imagen(self):
        """Pantalla inicial: invitación a abrir páginas y el botón «Agregar hojas»."""
        for key, canvas in self.view_canvases.items():
            canvas.delete('all')
            w, h = canvas.winfo_width(), canvas.winfo_height()
            if key == 'right':
                canvas.create_text(
                    w / 2,
                    h / 2,
                    text='Aquí verás la página limpia',
                    fill=C['muted'],
                    font=fuente(10),
                )
                continue
            canvas.create_text(
                w / 2,
                h / 2 - 52,
                text='Abre una o varias páginas de manga para empezar',
                fill=C['text'],
                font=fuente(12, True),
            )
            canvas.create_window(w / 2, h / 2, window=self.add_button)
            canvas.create_text(
                w / 2,
                h / 2 + 48,
                text='o pulsa «Ejemplo» arriba para practicar',
                fill=C['muted'],
                font=fuente(9),
            )


def main():
    """Carga la fuente, crea la ventana de Tk con el tema de los controles y abre PanelClean."""
    registrar_fuentes()
    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.12)
    except tk.TclError:
        pass
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass
    style.configure(
        'TCombobox',
        fieldbackground=C['bg'],
        background=C['panel2'],
        foreground=C['text'],
        arrowcolor=C['accent'],
        bordercolor=C['line'],
        lightcolor=C['bg'],
        darkcolor=C['bg'],
        padding=6,
    )
    style.map(
        'TCombobox',
        fieldbackground=[('readonly', C['bg'])],
        selectbackground=[('readonly', C['bg'])],
        selectforeground=[('readonly', C['text'])],
    )
    style.configure(
        'Vertical.TScrollbar',
        background=C['panel2'],
        troughcolor=C['bg'],
        arrowcolor=C['muted'],
        bordercolor=C['bg'],
        lightcolor=C['panel2'],
        darkcolor=C['panel2'],
    )
    root.option_add('*TCombobox*Listbox.background', C['bg'])
    root.option_add('*TCombobox*Listbox.foreground', C['text'])
    root.option_add('*TCombobox*Listbox.selectBackground', C['select'])
    root.option_add('*TCombobox*Listbox.selectForeground', C['text'])
    root.option_add('*TCombobox*Listbox.font', FONT)
    PanelCleanApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
