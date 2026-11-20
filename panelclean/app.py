"""PanelClean 0.3: editor local de limpieza asistida de globos de manga."""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

from . import __version__
from .vision import (
    Region,
    Options,
    bgr_from_file,
    write_png,
    bounded_region,
    detect_bubbles,
    generate_mask,
    describe_background,
    clean_region,
    mask_whole,
    inner_area,
)

ROOT = Path(__file__).resolve().parent.parent
C = {
    'bg': '#090f1a',
    'top': '#0c1524',
    'panel': '#121e30',
    'panel2': '#18273b',
    'line': '#283b51',
    'text': '#edf4fa',
    'muted': '#8fa4bb',
    'cyan': '#43d8c6',
    'cyan2': '#16434a',
    'orange': '#f4b96e',
    'red': '#fb7a88',
    'canvas': '#0b1421',
    'select': '#193c46',
    'white': '#ffffff',
    'good': '#65dcc6',
}
FONT = ('Segoe UI', 10)
SMALL = ('Segoe UI', 9)


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
            bg=C['panel2'],
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


class PanelCleanApp:
    """Ventana principal de PanelClean.

    Estado principal:
    - original: página tal como se abrió (nunca se modifica).
    - working: resultado que se muestra a la derecha; se recompone desde original.
    - regions: rectángulos marcados, cada uno con su máscara.
    - applied_ids: ids de las regiones cuya limpieza está aplicada en working.
    - undo_stack: pasos anteriores para «Deshacer».
    """

    def __init__(self, root: tk.Tk):
        """Crea el estado vacío, construye la interfaz y carga la página de ejemplo."""
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
        self.show_masks = tk.BooleanVar(value=True)
        self.threshold = tk.StringVar(value='Otsu')
        self.method = tk.StringVar(value='Automático')
        self.tool = tk.StringVar(value='Seleccionar')
        self.sensitivity = tk.IntVar(value=4)
        self.expansion = tk.IntVar(value=1)
        self.radius = tk.IntVar(value=3)
        self.brush = tk.IntVar(value=5)
        self.zoom = 1.0
        self.panx, self.pany = 0.0, 0.0
        self._preview_img = {}
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
        root.bind('<Destroy>', self._on_destroy, add='+')
        root.after_idle(self.load_demo)

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
        """Entrega un id de región que nunca se ha usado en esta página."""
        new_id = self.next_id
        self.next_id += 1
        return new_id

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
            font=('Segoe UI', size, 'bold' if bold else 'normal'),
            **kwargs,
        )

    def _button(self, parent, text, command, kind='normal', help_text=None):
        """Crea un botón plano que cambia de color bajo el cursor; kind='primary' lo destaca."""
        is_main = kind == 'primary'
        bg = C['cyan'] if is_main else C['panel2']
        fg = C['top'] if is_main else C['text']
        hover = '#78f2df' if is_main else '#263a52'
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            highlightthickness=0,
            bd=0,
            relief='flat',
            cursor='hand2',
            font=('Segoe UI', 10, 'bold' if is_main else 'normal'),
            padx=12,
            pady=8,
        )
        btn.bind('<Enter>', lambda _: btn.configure(bg=hover), add='+')
        btn.bind('<Leave>', lambda _: btn.configure(bg=bg), add='+')
        if help_text:
            Tooltip(btn, help_text)
        return btn

    def _header(self):
        """Construye la barra superior: logo, versión y botones Ejemplo, Abrir imagen y Exportar."""
        top = tk.Frame(self.root, bg=C['top'], height=72)
        top.pack(fill='x')
        top.pack_propagate(False)
        logo = tk.Frame(top, bg=C['cyan'], width=37, height=37)
        logo.pack(side='left', padx=(20, 11), pady=17)
        logo.pack_propagate(False)
        tk.Label(
            logo, text='P', bg=C['cyan'], fg=C['top'], font=('Segoe UI', 17, 'bold')
        ).pack(expand=True)
        brand = tk.Frame(top, bg=C['top'])
        brand.pack(side='left')
        tk.Label(
            brand,
            text='PanelClean',
            bg=C['top'],
            fg=C['text'],
            font=('Segoe UI', 17, 'bold'),
        ).pack(anchor='w')
        tk.Label(
            brand,
            text=f'Editor de globos · {__version__}',
            bg=C['top'],
            fg=C['muted'],
            font=SMALL,
        ).pack(anchor='w')
        self._button(
            top,
            'Exportar',
            self.save_image,
            'primary',
            'Guarda la página limpia y, junto a ella, la máscara y un reporte JSON.',
        ).pack(side='right', padx=(0, 17), pady=14)
        self._button(top, 'Abrir imagen', self.open_image).pack(
            side='right', padx=7, pady=14
        )
        self._button(top, 'Ejemplo', self.load_demo).pack(side='right', padx=7, pady=14)

    def _group(self, parent, title, pady=0):
        """Agrega un título de sección al panel y devuelve el contenedor de esa sección."""
        frame = tk.Frame(parent, bg=C['panel'])
        frame.pack(fill='x', padx=17, pady=(16, pady))
        self._label(frame, title, 10, 'text', True).pack(anchor='w')
        return frame

    def _scale(self, parent, title, var, lo, hi, callback=None):
        """Crea un control deslizante con su valor visible; callback se llama al moverlo."""
        line = tk.Frame(parent, bg=C['panel'])
        line.pack(fill='x', pady=(12, 1))
        self._label(line, title, 9, 'muted').pack(side='left')
        digit = self._label(line, str(var.get()), 9, 'cyan', True)
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
            bg=C['panel'],
            fg=C['muted'],
            highlightthickness=0,
            troughcolor=C['panel2'],
            activebackground=C['cyan'],
            bd=0,
            showvalue=False,
            sliderlength=15,
        )
        slider.pack(fill='x', pady=(0, 1))
        return slider

    def _combo(self, parent, title, variable, choices, callback):
        """Crea una lista desplegable de solo lectura; callback se llama al elegir una opción."""
        self._label(parent, title, 9, 'muted').pack(anchor='w', pady=(11, 5))
        cb = ttk.Combobox(
            parent, values=choices, textvariable=variable, state='readonly', font=FONT
        )
        cb.pack(fill='x')
        cb.bind('<<ComboboxSelected>>', lambda _: callback())
        return cb

    def _build_left(self, parent):
        """Construye el panel izquierdo (con barra de desplazamiento): lista de regiones,
        herramientas de edición de máscara y ajustes avanzados.
        """
        shell = tk.Frame(parent, bg=C['panel'])
        panel_canvas = tk.Canvas(shell, bg=C['panel'], highlightthickness=0, bd=0)
        scroll = tk.Scrollbar(
            shell,
            orient='vertical',
            command=panel_canvas.yview,
            bg=C['panel2'],
            troughcolor=C['panel'],
            activebackground=C['line'],
            width=10,
            bd=0,
            relief='flat',
            highlightthickness=0,
        )
        panel_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        panel_canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(panel_canvas, bg=C['panel'])
        window_id = panel_canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind(
            '<Configure>',
            lambda _: panel_canvas.configure(scrollregion=panel_canvas.bbox('all')),
        )
        panel_canvas.bind(
            '<Configure>',
            lambda e: panel_canvas.itemconfigure(window_id, width=e.width),
        )
        self.sidebar_scroll = panel_canvas
        self.sidebar_inner = inner

        self._group(inner, 'Regiones')
        self._label(
            inner,
            'Selecciona solo el área de las letras.',
            9,
            'muted',
            wraplength=230,
            justify='left',
        ).pack(anchor='w', padx=17, pady=(8, 12))
        self._button(
            inner,
            'Detectar globos',
            self.auto_detect,
            help_text='Búsqueda experimental de contornos. Conviene revisar los candidatos antes de limpiar.',
        ).pack(fill='x', padx=17, pady=(0, 10))
        holder = tk.Frame(inner, bg=C['panel2'], height=165)
        holder.pack(fill='x', padx=17)
        holder.pack_propagate(False)
        self.reg_list = tk.Listbox(
            holder,
            selectmode=tk.EXTENDED,
            bg=C['panel2'],
            fg=C['text'],
            selectbackground=C['select'],
            selectforeground=C['cyan'],
            bd=0,
            highlightthickness=0,
            relief='flat',
            font=('Consolas', 10),
            activestyle='none',
            exportselection=False,
        )
        self.reg_list.pack(fill='both', expand=True, padx=7, pady=7)
        self.reg_list.bind('<<ListboxSelect>>', self.on_list_selection)
        row = tk.Frame(inner, bg=C['panel'])
        row.pack(fill='x', padx=17, pady=(8, 5))
        self._button(
            row,
            'Quitar',
            self.remove_region,
            help_text='Elimina las regiones seleccionadas y quita su limpieza del resultado. Se puede deshacer.',
        ).pack(side='left', fill='x', expand=True, padx=(0, 5))
        self._button(
            row,
            'Crear máscaras',
            self.make_selected_mask,
            help_text='Detecta candidatos de texto para todas las regiones seleccionadas.',
        ).pack(side='left', fill='x', expand=True)
        hint = self._label(inner, 'Ctrl / Shift + clic para elegir varias.', 9, 'muted')
        hint.pack(anchor='w', padx=17, pady=(3, 12))
        self._group(inner, 'Edición')
        tools = tk.Frame(inner, bg=C['panel'])
        tools.pack(fill='x', padx=17, pady=(8, 5))
        for name, label in [
            ('Seleccionar', 'Seleccionar'),
            ('Pintar', 'Añadir máscara'),
            ('Borrar', 'Borrar máscara'),
        ]:
            rb = tk.Radiobutton(
                tools,
                text=label,
                variable=self.tool,
                value=name,
                command=self.render_views,
                bg=C['panel'],
                fg=C['text'],
                selectcolor=C['panel2'],
                activebackground=C['panel'],
                activeforeground=C['cyan'],
                font=SMALL,
                anchor='w',
                highlightthickness=0,
                bd=0,
                relief='flat',
            )
            rb.pack(fill='x', pady=3)
        tk.Checkbutton(
            inner,
            text='Ver máscara',
            variable=self.show_masks,
            command=self.render_views,
            bg=C['panel'],
            fg=C['muted'],
            selectcolor=C['panel2'],
            activebackground=C['panel'],
            activeforeground=C['cyan'],
            font=SMALL,
            highlightthickness=0,
            bd=0,
            relief='flat',
        ).pack(anchor='w', padx=17, pady=(8, 9))

        adv_btn = self._button(inner, 'Ajustes avanzados  ▾', self.toggle_advanced)
        adv_btn.pack(fill='x', padx=17, pady=(8, 8))
        self.advanced_button = adv_btn
        advanced = tk.Frame(inner, bg=C['panel'])
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
        self._label(
            inner,
            'Pasa el cursor por los botones para ver ayuda.',
            9,
            'muted',
            wraplength=245,
            justify='left',
        ).pack(anchor='w', padx=17, pady=(12, 18))
        return shell

    def toggle_advanced(self):
        """Muestra u oculta los ajustes avanzados de la máscara."""
        if self._advanced_open:
            self.advanced_frame.pack_forget()
            self.advanced_button.configure(text='Ajustes avanzados  ▾')
        else:
            self.advanced_frame.pack(fill='x', padx=17)
            self.advanced_button.configure(text='Ajustes avanzados  ▴')
        self._advanced_open = not self._advanced_open

    def _build_right(self, parent):
        """Construye el panel derecho: datos de la región, método de limpieza, botones y estado."""
        inspector = tk.Frame(parent, bg=C['panel'])
        self._group(inspector, 'Selección')
        card = tk.Frame(inspector, bg=C['panel2'])
        card.pack(fill='x', padx=16, pady=(12, 5))
        self.info_name = self._label(card, 'Sin región', 12, 'cyan', True)
        self.info_name.pack(anchor='w', padx=11, pady=(13, 5))
        self.info_entropy = self._label(card, 'Entropía  —', 9, 'muted')
        self.info_entropy.pack(anchor='w', padx=11, pady=3)
        self.info_dark = self._label(card, 'Tono oscuro  —', 9, 'muted')
        self.info_dark.pack(anchor='w', padx=11, pady=3)
        self.info_pixels = self._label(card, 'Máscara  —', 9, 'muted')
        self.info_pixels.pack(anchor='w', padx=11, pady=(3, 12))
        self._group(inspector, 'Limpieza')
        settings = tk.Frame(inspector, bg=C['panel'])
        settings.pack(fill='x', padx=17, pady=(4, 3))
        self._combo(
            settings,
            'Método',
            self.method,
            ['Automático', 'Relleno local', 'Inpainting Telea', 'Inpainting NS'],
            self.method_changed,
        )
        self._scale(settings, 'Radio de inpainting', self.radius, 1, 8)
        self._button(
            inspector,
            'Limpiar selección',
            self.clean_selected,
            'primary',
            help_text='Limpia todas las regiones marcadas en la lista y conserva las limpiezas anteriores.',
        ).pack(fill='x', padx=17, pady=(11, 8))
        self._button(
            inspector,
            'Limpiar todas',
            self.clean_all,
            help_text='Intenta procesar todas las regiones; las dudosas quedan intactas para revisión.',
        ).pack(fill='x', padx=17, pady=(0, 8))
        self._button(inspector, 'Deshacer', self.undo).pack(fill='x', padx=17)
        self._group(inspector, 'Estado')
        self.status_text = self._label(
            inspector,
            'Lista para comenzar.',
            9,
            'muted',
            wraplength=240,
            justify='left',
        )
        self.status_text.pack(anchor='w', padx=17, pady=(8, 12))
        return inspector

    def _build_center(self, parent):
        """Construye las vistas ORIGINAL y RESULTADO y conecta el ratón:
        clic izquierdo para seleccionar o pintar, rueda para zoom y clic derecho para desplazar.
        """
        center = tk.Frame(parent, bg=C['bg'])
        nav = tk.Frame(center, bg=C['bg'])
        nav.pack(fill='x', padx=4, pady=(2, 10))
        self._label(nav, 'Editor', 17, 'text', True).pack(side='left')
        self._button(nav, 'Ajustar vista', self.reset_view).pack(side='right')
        self._label(nav, 'Rueda: zoom  ·  clic derecho: mover', 9, 'muted').pack(
            side='right', padx=(0, 13)
        )
        boards = tk.Frame(center, bg=C['bg'])
        boards.pack(fill='both', expand=True)
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
            bar.pack(fill='x', padx=13, pady=(12, 11))
            self._label(bar, title, 9, 'cyan', True).pack(side='left')
            if key == 'left':
                self._label(bar, 'Selecciona texto, no el contorno', 9, 'muted').pack(
                    side='right'
                )
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
            cv.bind('<Button-4>', lambda e: self.adjust_zoom(1.12))
            cv.bind('<Button-5>', lambda e: self.adjust_zoom(1 / 1.12))
            if key == 'left':
                cv.bind('<ButtonPress-1>', self.left_down)
                cv.bind('<B1-Motion>', self.left_move)
                cv.bind('<ButtonRelease-1>', self.left_up)
            self.view_canvases[key] = cv
        return center

    def _build(self):
        """Arma la ventana completa: cabecera, tres paneles redimensionables y pie de página."""
        self._header()
        body = tk.PanedWindow(
            self.root,
            orient='horizontal',
            bg=C['bg'],
            bd=0,
            sashwidth=7,
            sashpad=0,
            opaqueresize=True,
            sashrelief='flat',
            showhandle=False,
        )
        body.pack(fill='both', expand=True, padx=10, pady=11)
        self.panes = body
        self.left_panel = self._build_left(body)
        self.center_panel = self._build_center(body)
        self.right_panel = self._build_right(body)
        body.add(self.left_panel, minsize=215, width=272)
        body.add(self.center_panel, minsize=495, stretch='always')
        body.add(self.right_panel, minsize=215, width=263)
        footer = tk.Frame(self.root, bg=C['top'], height=30)
        footer.pack(fill='x')
        footer.pack_propagate(False)
        self.message = self._label(
            footer, 'Selecciona las letras y crea una máscara.', 9, 'muted'
        )
        self.message.pack(side='left', padx=16, pady=6)
        self._label(footer, 'PanelClean  ·  TEL-328', 9, 'muted').pack(
            side='right', padx=17
        )
        self.root.bind_all('<MouseWheel>', self.sidebar_wheel, add='+')
        self.root.bind_all('<Button-4>', self.sidebar_wheel, add='+')
        self.root.bind_all('<Button-5>', self.sidebar_wheel, add='+')

    def sidebar_wheel(self, event):
        """Desplaza el panel izquierdo con la rueda cuando el cursor está sobre él."""
        widget = event.widget
        while widget is not None:
            if widget is self.sidebar_scroll or widget is self.sidebar_inner:
                if hasattr(event, 'num') and event.num in (4, 5):
                    direction = -1 if event.num == 4 else 1
                else:
                    direction = -1 if event.delta > 0 else 1
                self.sidebar_scroll.yview_scroll(direction * 3, 'units')
                return 'break'
            widget = getattr(widget, 'master', None)

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
        """Pide un archivo de imagen y lo abre."""
        path = filedialog.askopenfilename(
            title='Abrir página de manga',
            filetypes=[
                ('Imágenes', '*.png *.jpg *.jpeg *.bmp *.webp'),
                ('Todos', '*.*'),
            ],
        )
        if path:
            self.load_image(path)

    def load_image(self, path):
        """Carga una página y reinicia regiones, limpiezas, historial y vista."""
        try:
            image = bgr_from_file(str(path))
            if max(image.shape[:2]) > 3500 and not messagebox.askyesno(
                'Imagen grande',
                'Puede tardar más en procesarse. ¿Abrir de todas formas?',
            ):
                return
            self.original = image
            self.working = image.copy()
            self.source_path = str(path)
            self.regions = []
            self.selected_indices = set()
            self.focus_index = None
            self.applied_ids = set()
            self.undo_stack = []
            self.next_id = 1
            self.zoom = 1.0
            self.panx = self.pany = 0.0
            self.refresh_regions()
            self.render_views()
            self._set_status(
                f'Abierta: {Path(path).name}  ·  {image.shape[1]} × {image.shape[0]} px'
            )
        except Exception as exc:
            messagebox.showerror('No se pudo abrir la imagen', str(exc))

    def load_demo(self):
        """Abre la página de ejemplo incluida en examples/."""
        path = ROOT / 'examples' / 'demo_pagina.png'
        if path.exists():
            self.load_image(path)

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
        found = detect_bubbles(self.original)
        old_selection = set(self.selected_indices)
        new = []
        for candidate in found:
            if any(self._rect_overlap(candidate, r) > 0.5 for r in self.regions):
                continue
            candidate.id = self._new_id()
            self.regions.append(candidate)
            new.append(len(self.regions) - 1)
        self.selected_indices = old_selection | set(new)
        if new:
            self.focus_index = new[-1]
        self.refresh_regions()
        self.render_views()
        self._set_status(
            f'{len(new)} posibles globos añadidos. Selecciona las letras dentro de cada globo para mayor precisión.'
        )

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
            self.reg_list.insert('end', f' {state}  R{reg.id:02d}   {reg.w}×{reg.h}')
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
            text=f'R{r.id:02d}  ·  {r.classification}',
            fg=C['cyan'] if not r.review_reason else C['orange'],
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
        """Elimina las regiones seleccionadas y recompone el resultado sin ellas. Se puede deshacer."""
        if not self.selected_indices:
            return
        selected = self.selected_regions()
        self._push_undo(self.working.copy(), self.applied_ids, self.regions)
        for r in selected:
            self.applied_ids.discard(r.id)
        self.regions = [
            r for i, r in enumerate(self.regions) if i not in self.selected_indices
        ]
        self.selected_indices = set()
        self.focus_index = None
        self._rebuild_without_snapshot()
        self.refresh_regions()
        self.render_views()
        self._set_status(
            f'{len(selected)} región(es) eliminada(s). Se actualizó el resultado; puedes deshacerlo.'
        )

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
                outline=C['orange'],
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
                self.add_region((start[0], start[1], point[0], point[1]))
            except ValueError as exc:
                self._set_status(str(exc))

    def add_region(self, xyxy):
        """Crea una región con el rectángulo (x1, y1, x2, y2), la selecciona y la devuelve.

        Lanza ValueError si el rectángulo queda menor a 12 × 12 píxeles.
        """
        r = bounded_region(self.original, xyxy)
        r.id = self._new_id()
        self.regions.append(r)
        index = len(self.regions) - 1
        self.selected_indices.add(index)
        self.focus_index = index
        self.method.set('Automático')
        self.refresh_regions()
        self.render_views()
        self._set_status(
            f'R{r.id:02d} añadida. Se mantienen {len(self.selected_indices)} selecciones; crea las máscaras.'
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

    def _perform_clean(self, regions):
        """Limpia las regiones dadas y guarda un paso de «Deshacer».

        Las regiones que quedan en «Revisión manual» o sin máscara no se aplican. El
        resultado se recompone siempre desde el original, para no aplicar inpainting
        sobre zonas ya limpiadas.
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
                    skipped.append(f'R{r.id:02d}')
                else:
                    if (
                        opts.method != 'Automático'
                        and r.classification == 'Revisar máscara'
                    ):
                        confirmed = messagebox.askyesno(
                            'Confirmar limpieza',
                            f'La máscara de R{r.id:02d} ocupa demasiado espacio y podría incluir el dibujo.\n\n'
                            '¿Deseas aplicar el método manual de todas formas?',
                        )
                        if not confirmed:
                            self.applied_ids.discard(r.id)
                            skipped.append(f'R{r.id:02d}')
                            continue
                    self.applied_ids.add(r.id)
                    r.applied_radius = opts.radius
            self._rebuild_without_snapshot()
            # 3) Resumen para la barra de estado.
            for r in regions:
                if r.id in self.applied_ids:
                    applied += 1
                    methods.append(r.last_method)
                elif f'R{r.id:02d}' not in skipped:
                    skipped.append(f'R{r.id:02d}')
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
        self.refresh_regions()
        self.render_views()
        self._set_status('Limpieza deshecha. Las regiones y máscaras siguen editables.')

    def save_image(self):
        """Exporta la página limpia (PNG), la máscara usada (_mascara.png) y un reporte (_reporte.json)."""
        if self.working is None:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension='.png',
            initialfile='panelclean_resultado.png',
            filetypes=[('Imagen PNG', '*.png')],
            title='Exportar página limpia',
        )
        if not dest:
            return
        try:
            write_png(dest, self.working)
            mask = mask_whole(
                self.original, [r for r in self.regions if r.id in self.applied_ids]
            )
            stem = str(Path(dest).with_suffix(''))
            write_png(stem + '_mascara.png', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            info = {
                'proyecto': 'PanelClean',
                'version': __version__,
                'origen': Path(self.source_path).name,
                'regiones': [
                    {
                        'id': r.id,
                        'rect': r.rect,
                        'procesada': r.id in self.applied_ids,
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
                    for r in self.regions
                ],
            }
            Path(stem + '_reporte.json').write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding='utf8'
            )
            self._set_status('Se exportaron la imagen, máscara y reporte.')
            messagebox.showinfo(
                'Exportación completada',
                f'Guardados:\n{Path(dest).name}\n{Path(stem+"_mascara.png").name}\n{Path(stem+"_reporte.json").name}',
            )
        except Exception as exc:
            messagebox.showerror('Error al exportar', str(exc))

    def reset_view(self):
        """Vuelve al zoom inicial y centra las vistas."""
        self.panx = self.pany = 0.0
        self.zoom = 1.0
        self.render_views()

    def adjust_zoom(self, factor):
        """Multiplica el zoom por factor, manteniéndolo entre 0,65× y 6×."""
        self.zoom = max(0.65, min(6.0, self.zoom * factor))
        self.render_views()

    def mouse_zoom(self, event):
        """Acerca o aleja con la rueda del ratón (Windows y macOS; Linux usa los botones 4 y 5)."""
        self.adjust_zoom(1.13 if event.delta > 0 else 1 / 1.13)

    def pan_start(self, event):
        """Guarda el punto donde empieza el desplazamiento con el botón derecho."""
        self._pan_anchor = (event.x, event.y)

    def pan_move(self, event):
        """Desplaza ambas vistas siguiendo el ratón."""
        if self._pan_anchor is None:
            return
        ax, ay = self._pan_anchor
        self.panx += event.x - ax
        self.pany += event.y - ay
        self._pan_anchor = (event.x, event.y)
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

    def render_views(self):
        """Dibuja la página original (con máscaras y rectángulos) y el resultado.

        Guarda en _mapping la escala y el desplazamiento de cada vista, que
        _to_image_xy usa para convertir clics a coordenadas de la imagen.
        """
        if self.original is None:
            return
        for key, image in [('left', self.original), ('right', self.working)]:
            if image is None:
                continue
            canvas = self.view_canvases[key]
            w, h = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
            ih, iw = image.shape[:2]
            # Escala que hace caber la página en la vista (con 11 px de margen por
            # lado), multiplicada por el zoom; luego se centra y se suma el desplazamiento.
            scale = min((w - 22) / iw, (h - 22) / ih) * self.zoom
            tw, th = max(1, int(iw * scale)), max(1, int(ih * scale))
            ox = (w - tw) / 2 + self.panx
            oy = (h - th) / 2 + self.pany
            self._mapping[key] = (scale, ox, oy)
            pic = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert(
                'RGBA'
            )
            if key == 'left' and self.show_masks.get():
                # Máscaras en rosa semitransparente (alfa 104 de 255, cerca de 40 %).
                overlay = Image.new('RGBA', pic.size, (0, 0, 0, 0))
                for r in self.regions:
                    if r.mask is not None:
                        alpha = Image.fromarray(
                            (r.mask > 0).astype(np.uint8) * 104, 'L'
                        )
                        tint = Image.new('RGBA', (r.w, r.h), (255, 71, 100, 0))
                        tint.putalpha(alpha)
                        overlay.alpha_composite(tint, (r.x, r.y))
                pic = Image.alpha_composite(pic, overlay)
            picture = pic.resize((tw, th), Image.Resampling.LANCZOS)
            self._preview_img[key] = ImageTk.PhotoImage(picture, master=self.root)
            canvas.delete('all')
            canvas.create_image(ox, oy, image=self._preview_img[key], anchor='nw')
            if key == 'left':
                for i, r in enumerate(self.regions):
                    selected = i in self.selected_indices
                    color = C['cyan'] if selected else C['orange']
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
                        text=f'R{r.id:02d}',
                        fill=C['top'],
                        font=('Segoe UI', 9, 'bold'),
                    )


def main():
    """Crea la ventana de Tk, aplica el tema de los controles y abre PanelClean."""
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
        fieldbackground=C['panel2'],
        background=C['panel2'],
        foreground=C['text'],
        arrowcolor=C['cyan'],
        padding=7,
    )
    style.map(
        'TCombobox',
        fieldbackground=[('readonly', C['panel2'])],
        selectbackground=[('readonly', C['panel2'])],
        selectforeground=[('readonly', C['text'])],
    )
    style.configure(
        'Vertical.TScrollbar',
        background=C['panel2'],
        troughcolor=C['panel'],
        arrowcolor=C['muted'],
        bordercolor=C['panel'],
    )
    root.option_add('*TCombobox*Listbox.background', C['panel2'])
    root.option_add('*TCombobox*Listbox.foreground', C['text'])
    PanelCleanApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
