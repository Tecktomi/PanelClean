"""Pruebas de la interfaz (panelclean/app.py) manejada por código, sin clics.

Cada prueba crea la ventana oculta, carga la página de ejemplo y llama a los
mismos métodos que usan los botones. Ejecutar desde la raíz del repositorio:
    python -m unittest discover tests
"""

import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from panelclean import app as A

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

    def test_los_ids_crecen_y_no_se_repiten(self):
        self.assertEqual([r.id for r in self.app.regions], [1, 2, 3])
        self.seleccionar(2)
        self.app.remove_region()
        nueva = self.app.add_region(TEXTOS[2])
        self.assertEqual(nueva.id, 4)

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
