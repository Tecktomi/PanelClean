"""Pruebas de la interfaz (panelclean/app.py) manejada por código, sin clics.

Cada prueba crea la ventana oculta, carga la página de ejemplo y llama a los
mismos métodos que usan los botones. Ejecutar desde la raíz del repositorio:
    python -m unittest discover tests
"""

import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from panelclean import app as A

MANGA = A.ROOT / 'datasets' / 'open-mantra' / 'images'
TEXTOS = [(415, 340, 590, 445), (1095, 165, 1290, 260), (1120, 635, 1265, 740)]


class AppTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:  # sin pantalla (por ejemplo, un servidor)
            self.skipTest(f'Tk no disponible: {exc}')
        self.root.withdraw()
        self.app = A.PanelCleanApp(self.root)
        self.root.update()
        self.app.load_demo()
        for caja in TEXTOS:
            self.app.add_region(caja)

    def tearDown(self):
        self.root.destroy()

    def seleccionar(self, *indices):
        self.app.selected_indices = set(indices)
        self.app.focus_index = indices[-1] if indices else None

    def recorte(self, region):
        return self.app.working[
            region.y : region.y + region.h, region.x : region.x + region.w
        ].copy()

    # --- flujo principal -------------------------------------------------
    def test_flujo_completo_y_exportacion(self):
        self.seleccionar(0, 1, 2)
        self.app.make_selected_mask()
        self.app.clean_selected()
        self.assertEqual(self.app.applied_ids, {1, 2, 3})
        self.assertFalse(np.array_equal(self.app.working, self.app.original))
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp) / 'pagina.png'
            with mock.patch.object(
                A.filedialog, 'asksaveasfilename', return_value=str(destino)
            ), mock.patch.object(A.messagebox, 'showinfo'):
                self.app.save_image()
            for nombre in ('pagina.png', 'pagina_mascara.png', 'pagina_reporte.json'):
                self.assertTrue((Path(tmp) / nombre).exists(), nombre)

    def test_los_nombres_reusan_numeros_libres_y_los_ids_no_se_repiten(self):
        # El id interno (lo usa «Deshacer») nunca se repite; el nombre visible
        # (R01, R02…) toma el menor número libre, así no quedan saltos.
        self.assertEqual([r.id for r in self.app.regions], [1, 2, 3])
        self.seleccionar(1)
        self.app.remove_region()
        nueva = self.app.add_region(TEXTOS[1])
        self.assertEqual(nueva.id, 4)
        self.assertEqual(self.app._nombre(nueva), 'R02')
        self.app.remove_all()
        self.assertEqual(self.app._nombre(self.app.add_region(TEXTOS[0])), 'R01')

    def test_quitar_quita_solo_la_region_elegida(self):
        self.seleccionar(0, 2)  # con foco en la última elegida
        self.app.remove_region()
        self.assertEqual(
            [self.app._nombre(r) for r in self.app.regions], ['R01', 'R02']
        )

    def test_quitar_todo_se_puede_deshacer(self):
        self.seleccionar(0, 1, 2)
        self.app.clean_selected()
        limpia = self.app.working.copy()
        self.app.remove_all()
        self.assertEqual(self.app.regions, [])
        self.assertTrue(np.array_equal(self.app.working, self.app.original))
        self.app.undo()
        self.assertEqual(len(self.app.regions), 3)
        self.assertTrue(np.array_equal(self.app.working, limpia))

    def test_paneles_se_ocultan_y_se_muestran(self):
        self.root.deiconify()
        self.app.toggle_panel('left')
        self.root.update()
        self.assertFalse(self.app.left_panel.winfo_ismapped())
        self.app.toggle_panel('left')
        self.root.update()
        self.assertTrue(self.app.left_panel.winfo_ismapped())

    def test_sin_imagen_muestra_el_boton_agregar(self):
        ventana = tk.Toplevel(self.root)
        app = A.PanelCleanApp(ventana)
        ventana.update()
        app.render_views()
        lienzo = app.view_canvases['left']
        self.assertIsNone(app.original)
        self.assertIn('window', [lienzo.type(i) for i in lienzo.find_all()])
        ventana.destroy()

    # --- zona aproximada ---------------------------------------------------
    def test_una_zona_holgada_se_ajusta_al_texto(self):
        self.app.remove_all()
        x1, y1, x2, y2 = TEXTOS[0]
        zona = (x1 - 40, y1 - 40, x2 + 40, y2 + 40)
        nuevas = self.app.add_zone(zona)
        self.assertEqual(len(nuevas), 1)
        r = nuevas[0]
        self.assertLess(r.w * r.h, 0.6 * (zona[2] - zona[0]) * (zona[3] - zona[1]))
        self.assertEqual(self.app._nombre(r), 'R01')
        self.assertEqual(self.app.add_zone(zona), [], 'no repite regiones')
        self.app.ajustar.set(False)
        tal_cual = self.app.add_zone(zona)[0]
        self.assertEqual(tal_cual.rect, (zona[0], zona[1], 255, 185))

    # --- cola de hojas ----------------------------------------------------
    def abrir_cola(self, *extra):
        paginas = [A.ROOT / 'examples' / 'demo_pagina.png'] + [
            MANGA / libro / 'ja' / '005.jpg'
            for libro in ('rasetugari', 'tojime_no_siora')
        ]
        self.assertEqual(self.app.abrir_hojas(paginas + list(extra)), 3 + len(extra))

    def procesar_cola(self):
        # Como se espera que se use: el texto del ejemplo se encierra a mano; las
        # demás hojas, sin regiones, usan la búsqueda automática de globos.
        for x1, y1, x2, y2 in TEXTOS:
            self.app.add_zone((x1 - 30, y1 - 30, x2 + 30, y2 + 30))
        with mock.patch.object(A.messagebox, 'askyesnocancel', return_value=True):
            self.app.procesar_todas()
            limite = time.time() + 120
            while self.app._procesando and time.time() < limite:
                self.root.update()
        self.assertFalse(self.app._procesando, 'la cola debe terminar')

    def test_cada_hoja_conserva_sus_regiones(self):
        self.abrir_cola()
        self.assertTrue(
            self.app.tira_hojas.winfo_manager(), 'con varias hojas se ve la tira'
        )
        self.app.add_region(TEXTOS[0])
        self.app.ir_a_hoja(1)
        self.assertEqual(self.app.regions, [])
        self.app.add_region((100, 100, 200, 200))
        self.app.ir_a_hoja_relativa(-1)
        self.assertEqual(self.app.actual, 0)
        self.assertEqual([r.rect for r in self.app.regions], [(415, 340, 175, 105)])
        self.app.ir_a_hoja(1)
        self.assertEqual(len(self.app.regions), 1)

    def test_procesar_todas_limpia_cada_hoja(self):
        self.abrir_cola()
        self.procesar_cola()
        for hoja in self.app.hojas:
            self.assertTrue(hoja.procesada, hoja.nombre)
            self.assertIn(hoja.estado, ('Lista', 'Revisar'))
        ejemplo = self.app.hojas[0]
        self.assertTrue(ejemplo.applied_ids, 'el ejemplo debe quedar con texto limpio')
        self.assertFalse(np.array_equal(ejemplo.working, ejemplo.original))

    def test_exportar_todas_guarda_cada_hoja_sin_pisar_nombres(self):
        # La misma página dos veces: los nombres no deben repetirse.
        self.abrir_cola(A.ROOT / 'examples' / 'demo_pagina.png')
        self.procesar_cola()
        self.app.ir_a_hoja(3)
        self.procesar_cola()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                A.filedialog, 'askdirectory', return_value=tmp
            ), mock.patch.object(A.messagebox, 'showinfo'):
                self.app.exportar_todas()
            nombres = {p.name for p in Path(tmp).iterdir()}
        for nombre in ('demo_pagina_limpia.png', 'demo_pagina_limpia_2.png'):
            self.assertIn(nombre, nombres)
        exportadas = [h for h in self.app.hojas if h.applied_ids]
        self.assertEqual(
            len([n for n in nombres if n.endswith('_reporte.json')]), len(exportadas)
        )
        self.assertTrue(all(h.exportada for h in exportadas))

    def test_avisa_antes_de_cerrar_limpiezas_sin_exportar(self):
        self.seleccionar(0)
        self.app.clean_selected()
        with mock.patch.object(
            A.messagebox, 'askyesno', return_value=False
        ) as pregunta:
            self.assertEqual(
                self.app.load_image(MANGA / 'rasetugari' / 'ja' / '005.jpg'), 0
            )
        pregunta.assert_called_once()
        self.assertEqual(Path(self.app.source_path).name, 'demo_pagina.png')

    # --- vistas ----------------------------------------------------------
    def test_la_vista_no_se_recalcula_al_mover_pero_si_al_cambiar_la_mascara(self):
        self.app.render_views()
        base = self.app._pagina_de_vista('left')
        self.app.zoom, self.app.panx = 3.0, 40.0
        self.app.render_views()
        self.assertIs(self.app._pagina_de_vista('left'), base, 'debe usar la caché')
        self.seleccionar(0)
        self.app.make_selected_mask()
        con_mascara = self.app._pagina_de_vista('left')
        self.assertIsNot(con_mascara, base)
        r = self.app.regions[0]
        marcada = con_mascara[r.y : r.y + r.h, r.x : r.x + r.w][r.mask > 0]
        self.assertTrue((marcada[:, 0] > marcada[:, 2]).all(), 'la máscara va en rosa')
        self.app.tool.set('Pintar')
        self.app.paint_mask((r.x + 10, r.y + 10))
        self.assertIsNot(self.app._pagina_de_vista('left'), con_mascara)
        resultado = self.app._pagina_de_vista('right')
        self.app.clean_selected()
        self.assertIsNot(self.app._pagina_de_vista('right'), resultado)

    def test_el_zoom_con_la_rueda_deja_fijo_el_punto_bajo_el_cursor(self):
        self.app.render_views()
        lienzo = self.app.view_canvases['left']

        def punto_bajo(evento):
            escala, ox, oy = self.app._mapping['left']
            return (evento.x - ox) / escala, (evento.y - oy) / escala

        evento = mock.Mock(widget=lienzo, x=70, y=35, delta=120)
        antes = punto_bajo(evento)
        for _ in range(4):
            self.app.mouse_zoom(evento)
        self.assertGreater(self.app.zoom, 1.5)
        for a, b in zip(antes, punto_bajo(evento)):
            self.assertAlmostEqual(a, b, places=6)

    # --- errores corregidos ----------------------------------------------
    def test_quitar_se_puede_deshacer(self):
        self.seleccionar(0, 1)
        self.app.clean_selected()
        limpia = self.app.working.copy()
        self.seleccionar(0)
        self.app.remove_region()
        self.assertEqual([r.id for r in self.app.regions], [2, 3])
        self.assertNotIn(1, self.app.applied_ids)
        self.app.undo()
        self.assertEqual([r.id for r in self.app.regions], [1, 2, 3])
        self.assertEqual(self.app.applied_ids, {1, 2})
        self.assertTrue(np.array_equal(self.app.working, limpia))

    def test_cambiar_el_radio_no_altera_limpiezas_anteriores(self):
        trama = self.app.regions[2]
        self.seleccionar(2)
        self.app.clean_selected()
        antes = self.recorte(trama)
        self.app.radius.set(8)
        self.seleccionar(0)
        self.app.clean_selected()
        self.assertTrue(np.array_equal(self.recorte(trama), antes))

    def test_crear_mascaras_pregunta_antes_de_borrar_correcciones(self):
        self.seleccionar(0)
        self.app.make_selected_mask()
        r = self.app.regions[0]
        self.app.tool.set('Pintar')
        self.app.paint_mask((r.x + 10, r.y + 10))
        self.app.end_brush()
        pintada = r.mask.copy()
        with mock.patch.object(
            A.messagebox, 'askyesno', return_value=False
        ) as pregunta:
            self.app.make_selected_mask()
        pregunta.assert_called_once()
        self.assertTrue(np.array_equal(r.mask, pintada), 'debe conservar la corrección')
        with mock.patch.object(A.messagebox, 'askyesno', return_value=True):
            self.app.make_selected_mask()
        self.assertFalse(np.array_equal(r.mask, pintada), 'si acepta, se recalcula')

    def test_la_ventana_cabe_en_la_pantalla(self):
        self.root.deiconify()
        self.root.update()
        self.assertLessEqual(self.root.winfo_width(), self.root.winfo_screenwidth())
        self.assertLessEqual(self.root.winfo_height(), self.root.winfo_screenheight())


if __name__ == '__main__':
    unittest.main()
