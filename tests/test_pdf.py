"""Pruebas de la lectura de PDF (panelclean/pdf.py) y de su uso en la aplicación.

Los PDF se arman aquí mismo, con páginas de OpenMantra y con dibujos. Ejecutar desde
la raíz del repositorio:
    python -m unittest discover tests
"""

import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from panelclean import app as A
from panelclean import pdf
from panelclean import vision as V

try:
    import pypdfium2  # noqa: F401
except ImportError:
    pypdfium2 = None

MANGA = A.ROOT / 'datasets' / 'open-mantra' / 'images' / 'rasetugari' / 'ja'
PAGINAS = [MANGA / f'{n:03d}.jpg' for n in (3, 5, 8)]


def pdf_de_imagenes(destino: Path, imagenes: list[Image.Image]) -> Path:
    """PDF con una imagen por página, como un manga escaneado."""
    imagenes[0].save(destino, save_all=True, append_images=imagenes[1:])
    return destino


def pdf_a_mano(destino: Path, contenido: bytes, recursos=b'', extra=()) -> Path:
    """PDF de una página de 200 × 300 puntos con el contenido dado."""
    objetos = [
        b'<</Type/Catalog/Pages 2 0 R>>',
        b'<</Type/Pages/Kids[3 0 R]/Count 1>>',
        b'<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 300]/Contents 4 0 R'
        b'/Resources<<' + recursos + b'>>>>',
        b'<</Length %d>>stream\n%s\nendstream' % (len(contenido), contenido),
        *extra,
    ]
    datos, inicios = b'%PDF-1.4\n', []
    for n, objeto in enumerate(objetos, 1):
        inicios.append(len(datos))
        datos += b'%d 0 obj%s endobj\n' % (n, objeto)
    tabla = b''.join(b'%010d 00000 n \n' % i for i in inicios)
    destino.write_bytes(
        datos
        + b'xref\n0 %d\n0000000000 65535 f \n%s' % (len(objetos) + 1, tabla)
        + b'trailer<</Size %d/Root 1 0 R>>startxref\n%d\n%%%%EOF'
        % (len(objetos) + 1, len(datos))
    )
    return destino


def imagen_pdf(pixeles: np.ndarray) -> bytes:
    """Objeto de PDF con una imagen en grises, sin comprimir."""
    alto, ancho = pixeles.shape
    return (
        b'<</Type/XObject/Subtype/Image/Width %d/Height %d/ColorSpace/DeviceGray'
        b'/BitsPerComponent 8/Length %d>>stream\n%s\nendstream'
        % (ancho, alto, pixeles.size, pixeles.tobytes())
    )


class NombresTest(unittest.TestCase):
    """Lo que no necesita la biblioteca para leer PDF."""

    def test_nombres_de_paginas(self):
        ruta = Path('C:/mis mangas/Tomo #1 (ñ).PDF#012')
        self.assertEqual(pdf.separar(ruta), (Path('C:/mis mangas/Tomo #1 (ñ).PDF'), 12))
        self.assertEqual(pdf.nombre_base(ruta), 'Tomo #1 (ñ)_p012')
        self.assertFalse(pdf.es_pdf(ruta))
        self.assertTrue(pdf.es_pdf('C:/mis mangas/Tomo #1 (ñ).PDF'))
        for suelta in ('C:/x/hoja.png', 'C:/x/a#3.png', 'C:/x/tomo.pdf'):
            self.assertIsNone(pdf.separar(suelta))
        self.assertEqual(pdf.nombre_base('C:/x/hoja.v2.png'), 'hoja.v2')

    def test_elegir_paginas(self):
        self.assertEqual(pdf.elegir('3, 8, 12-14', 20), [3, 8, 12, 13, 14])
        self.assertEqual(pdf.elegir(' 2 - 4 ; 3', 20), [2, 3, 4], 'no repite')
        for malo in ('', 'a', '0', '5-2', '1-21', '1,,3', '-3', '1.5'):
            with self.assertRaises(ValueError, msg=malo):
                pdf.elegir(malo, 20)

    def test_sin_la_biblioteca_se_explica_que_falta(self):
        with mock.patch.dict('sys.modules', {'pypdfium2': None}):
            with self.assertRaisesRegex(OSError, 'pypdfium2'):
                pdf.paginas('tomo.pdf')


@unittest.skipUnless(pypdfium2, 'pypdfium2 no está instalado')
class LecturaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.carpeta = Path(self.tmp.name)

    def test_un_manga_escaneado_se_lee_en_su_tamano_y_en_orden(self):
        originales = [V.bgr_from_file(str(p)) for p in PAGINAS]
        tomo = pdf_de_imagenes(
            self.carpeta / 'año ñ.pdf', [Image.open(p).convert('RGB') for p in PAGINAS]
        )
        rutas = pdf.paginas(tomo)
        self.assertEqual(
            [r.name for r in rutas][0::2], ['año ñ.pdf#001', 'año ñ.pdf#003']
        )
        for ruta, original in zip(rutas, originales, strict=True):
            leida = V.bgr_from_file(str(ruta))
            self.assertEqual(leida.shape, original.shape)
            # El PDF vuelve a comprimir en JPEG: se parecen, no son idénticas.
            self.assertLess(np.abs(leida.astype(int) - original).mean(), 4)

    def test_en_grises_sin_perdida_es_identica(self):
        pixeles = np.random.default_rng(0).integers(0, 256, (90, 60), dtype=np.uint8)
        # Ocupa solo el centro de la página: los márgenes del PDF no se leen.
        tomo = pdf_a_mano(
            self.carpeta / 'gris.pdf',
            b'q 120 0 0 180 40 60 cm /I Do Q',
            b'/XObject<</I 5 0 R>>',
            [imagen_pdf(pixeles)],
        )
        leida = pdf.leer(tomo, 1)
        self.assertTrue(
            np.array_equal(leida, cv2.cvtColor(pixeles, cv2.COLOR_GRAY2BGR))
        )

    def test_el_texto_invisible_de_un_ocr_no_cuenta(self):
        pixeles = np.random.default_rng(1).integers(0, 256, (90, 60), dtype=np.uint8)
        recursos = (
            b'/XObject<</I 5 0 R>>'
            b'/Font<</F<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>'
        )
        for modo, tal_cual in ((3, True), (0, False)):
            tomo = pdf_a_mano(
                self.carpeta / 'ocr.pdf',
                b'q 200 0 0 300 0 0 cm /I Do Q '
                b'BT %d Tr /F 40 Tf 20 100 Td (hola) Tj ET' % modo,
                recursos,
                [imagen_pdf(pixeles)],
            )
            leida = pdf.leer(tomo, 1)
            identica = np.array_equal(leida, cv2.cvtColor(pixeles, cv2.COLOR_GRAY2BGR))
            self.assertEqual(identica, tal_cual)

    def test_una_pagina_con_algo_mas_se_dibuja_completa(self):
        pixeles = np.full((90, 60), 200, dtype=np.uint8)
        recursos, extra = b'/XObject<</I 5 0 R>>', [imagen_pdf(pixeles)]
        casos = {
            'con un rectángulo encima': b'q 200 0 0 300 0 0 cm /I Do Q 0 g 0 0 50 50 re f',
            'girada': b'q 0 -300 200 0 0 300 cm /I Do Q',
            'dos imágenes': b'q 200 0 0 150 0 0 cm /I Do Q q 200 0 0 150 0 150 cm /I Do Q',
            'sin imágenes': b'0 g 0 0 50 50 re f',
        }
        for nombre, contenido in casos.items():
            with self.subTest(nombre):
                tomo = pdf_a_mano(
                    self.carpeta / 'mixta.pdf', contenido, recursos, extra
                )
                leida = pdf.leer(tomo, 1)
                alto, ancho = leida.shape[:2]
                self.assertEqual(leida.ndim, 3)
                self.assertAlmostEqual(alto / ancho, 1.5, delta=0.05)
        # El rectángulo negro de abajo a la izquierda se ve.
        self.assertTrue((leida[-5, 5] < 10).all())
        self.assertEqual(leida.shape[0], round(300 * pdf.PPP / 72 + 0.49))

    def test_una_pagina_dibujada_no_pasa_del_lado_maximo(self):
        with mock.patch.object(pdf, 'LADO_MAXIMO', 150):
            tomo = pdf_a_mano(self.carpeta / 'v.pdf', b'0 g 0 0 50 50 re f')
            self.assertLessEqual(max(pdf.leer(tomo, 1).shape), 151)

    def test_los_que_no_se_pueden_leer_dicen_por_que(self):
        falso = self.carpeta / 'falso.pdf'
        falso.write_bytes(b'esto no es un PDF')
        with self.assertRaisesRegex(OSError, 'no se pudo leer como PDF'):
            pdf.paginas(falso)
        tomo = pdf_a_mano(self.carpeta / 'v.pdf', b'0 g 0 0 50 50 re f')
        with self.assertRaisesRegex(ValueError, 'tiene 1 páginas'):
            pdf.leer(tomo, 2)
        with self.assertRaises(FileNotFoundError):
            pdf.paginas(self.carpeta / 'no_existe.pdf')
        # No queda abierto: en Windows no se podría borrar.
        tomo.unlink()


@unittest.skipUnless(pypdfium2, 'pypdfium2 no está instalado')
class AplicacionTest(unittest.TestCase):
    """Abrir y exportar páginas de un PDF en la aplicación."""

    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Tk no disponible: {exc}')
        self.root.withdraw()
        self.app = A.PanelCleanApp(self.root)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tomo = pdf_de_imagenes(
            Path(self.tmp.name) / 'tomo.pdf',
            [Image.open(p).convert('RGB') for p in PAGINAS],
        )

    def tearDown(self):
        self.root.update()
        self.root.destroy()

    def abrir(self, *respuestas, rutas=None, **opciones):
        """Abre el PDF contestando la pregunta de qué páginas."""
        with mock.patch.object(
            A.simpledialog, 'askstring', side_effect=respuestas
        ) as self.pregunta:
            return self.app.abrir_hojas(rutas or [self.tomo], **opciones)

    def nombres(self):
        return [h.nombre for h in self.app.hojas]

    def test_abre_las_paginas_elegidas(self):
        self.assertEqual(self.abrir('1, 3'), 2)
        self.assertEqual(self.nombres(), ['tomo.pdf#001', 'tomo.pdf#003'])
        self.assertEqual(
            self.app.original.shape, V.bgr_from_file(str(PAGINAS[0])).shape
        )

    def test_si_la_respuesta_no_sirve_pregunta_de_nuevo(self):
        self.assertEqual(self.abrir('1-9', '2'), 1)
        self.assertEqual(self.pregunta.call_count, 2)
        self.assertIn('entre 1 y 3', self.pregunta.call_args.args[1])
        self.assertEqual(self.nombres(), ['tomo.pdf#002'])

    def test_cancelar_deja_todo_como_estaba(self):
        self.app.load_demo()
        self.assertEqual(self.abrir(None), 0)
        self.assertEqual(self.nombres(), ['demo_pagina.png'])

    def test_se_suma_a_la_cola_junto_a_imagenes(self):
        self.app.load_demo()
        self.assertEqual(
            self.abrir('2-3', rutas=[PAGINAS[0], self.tomo], agregar=True), 3
        )
        self.assertEqual(
            self.nombres(),
            ['demo_pagina.png', '003.jpg', 'tomo.pdf#002', 'tomo.pdf#003'],
        )

    def test_avisa_si_van_a_ocupar_mucha_memoria(self):
        with mock.patch.object(A, 'MEMORIA_AVISO', 1), mock.patch.object(
            A.messagebox, 'askyesno', return_value=False
        ) as aviso:
            self.assertEqual(self.abrir('1-3'), 0)
        aviso.assert_called_once()
        self.assertIn('0,0 GB', aviso.call_args.args[1])

    def test_un_pdf_malo_se_informa_y_el_resto_se_abre(self):
        malo = Path(self.tmp.name) / 'malo.pdf'
        malo.write_bytes(b'nada')
        with mock.patch.object(A.messagebox, 'showerror') as error:
            self.assertEqual(self.abrir('1', rutas=[malo, self.tomo]), 1)
        self.assertIn('malo.pdf', error.call_args.args[1])
        self.assertEqual(self.nombres(), ['tomo.pdf#001'])

    def test_exporta_una_imagen_por_pagina(self):
        self.abrir('1-2')
        for indice in (0, 1):
            self.app.ir_a_hoja(indice)
            self.app.add_region((100, 100, 300, 250))
            region = self.app.regions[0]
            region.mask = np.full((region.h, region.w), 255, dtype=np.uint8)
            self.app.applied_ids = {region.id}
        with tempfile.TemporaryDirectory() as salida, mock.patch.object(
            A.filedialog, 'askdirectory', return_value=salida
        ), mock.patch.object(A.messagebox, 'showinfo'):
            self.app.exportar_todas()
            nombres = {p.name for p in Path(salida).iterdir()}
        for pagina in ('001', '002'):
            for cola in ('.png', '_mascara.png', '_reporte.json'):
                self.assertIn(f'tomo_p{pagina}_limpia{cola}', nombres)


if __name__ == '__main__':
    unittest.main()
