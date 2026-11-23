"""Pruebas del procesamiento de imagen (panelclean/vision.py).

Ejecutar desde la raíz del repositorio:
    python -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from panelclean import vision as V

DEMO = Path(__file__).resolve().parent.parent / 'examples' / 'demo_pagina.png'

# Rectángulos (x1, y1, x2, y2) ajustados al texto de los tres globos de la página de ejemplo.
TEXTO_LISO = (415, 340, 590, 445)  # «¿DÓNDE ESTÁS?», globo blanco
TEXTO_PUNTOS = (
    1095,
    165,
    1290,
    260,
)  # «NO PUEDE SER...», globo blanco sobre viñeta punteada
TEXTO_TRAMA = (1120, 635, 1265, 740)  # «¿QUÉ PASÓ?», globo con trama


def mascara_completa(imagen, region):
    """Devuelve la máscara de la región ubicada en una imagen del tamaño de la página."""
    total = np.zeros(imagen.shape[:2], np.uint8)
    total[region.y : region.y + region.h, region.x : region.x + region.w] = region.mask
    return total


class RegionTest(unittest.TestCase):
    def setUp(self):
        self.img = V.bgr_from_file(str(DEMO))

    def test_lee_la_pagina_de_ejemplo(self):
        self.assertEqual(self.img.shape, (980, 1400, 3))

    def test_ordena_y_recorta_las_coordenadas(self):
        r = V.bounded_region(self.img, (1450, 50, 1300, -20))
        self.assertEqual((r.x, r.y, r.x + r.w, r.y + r.h), (1300, 0, 1400, 50))

    def test_rechaza_regiones_demasiado_chicas(self):
        with self.assertRaises(ValueError):
            V.bounded_region(self.img, (10, 10, 20, 100))


class MascaraTest(unittest.TestCase):
    def setUp(self):
        self.img = V.bgr_from_file(str(DEMO))
        self.opts = V.Options()

    def test_la_mascara_cubre_texto_y_queda_dentro_de_la_region(self):
        r = V.bounded_region(self.img, TEXTO_LISO)
        mask = V.generate_mask(self.img, r, self.opts)
        self.assertEqual(mask.shape, (r.h, r.w))
        self.assertGreater(cv2.countNonZero(mask), 1000)
        # Los píxeles marcados deben ser oscuros en la imagen original (tinta, no fondo).
        gris = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)[
            r.y : r.y + r.h, r.x : r.x + r.w
        ]
        sin_dilatar = V.generate_mask(self.img, r, V.Options(expansion=0))
        self.assertLess(float(gris[sin_dilatar > 0].mean()), 110)

    def test_no_marca_lineas_largas(self):
        # Página sintética: texto y una línea horizontal que cruza toda la región.
        img = np.full((160, 320, 3), 255, np.uint8)
        cv2.putText(img, 'HOLA', (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
        cv2.line(img, (0, 125), (319, 125), (0, 0, 0), 3)
        r = V.bounded_region(img, (0, 0, 320, 160))
        mask = V.generate_mask(img, r, V.Options(expansion=0))
        self.assertGreater(
            cv2.countNonZero(mask[30:100]), 300, 'debe marcar las letras'
        )
        self.assertEqual(cv2.countNonZero(mask[118:133]), 0, 'no debe marcar la línea')

    def test_no_borra_lo_que_entra_por_el_borde(self):
        # Una sombra que entra por una esquina de la selección y sigue afuera no
        # es texto: rellenarla dejaría un parche blanco con el borde del rectángulo.
        img = np.full((200, 300, 3), 255, np.uint8)
        cv2.putText(img, 'HOLA', (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
        cv2.rectangle(img, (200, 140), (299, 199), (30, 30, 30), -1)
        r = V.bounded_region(img, (60, 60, 230, 160))
        mask = V.generate_mask(img, r, self.opts)
        self.assertEqual(cv2.countNonZero(mask[140 - 60 :, 200 - 60 :]), 0)
        self.assertGreater(
            cv2.countNonZero(mask[:70, :140]), 300, 'debe marcar las letras'
        )

    def test_una_letra_cortada_por_la_seleccion_se_sigue_borrando(self):
        # Con una selección ajustada que corta la parte de arriba de las letras,
        # lo que queda adentro sigue siendo casi todo letra y se marca.
        img = np.full((160, 320, 3), 255, np.uint8)
        cv2.putText(img, 'HOLA', (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
        r = V.bounded_region(img, (50, 64, 270, 120))
        mask = V.generate_mask(img, r, V.Options(expansion=0))
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[r.y : r.y + r.h, r.x : r.x + r.w]
        tinta = (gris < 128) & (V.inner_area(r) > 0)
        self.assertGreater(float((mask[tinta] > 0).mean()), 0.9)

    def test_separa_letras_de_rayas_y_tramas(self):
        # Pares sintéticos con fondo estructurado (semilla de desarrollo, no la de
        # la evaluación). Antes, las letras que tocaban una raya se descartaban.
        from evaluacion import metricas as M
        from evaluacion import sinteticos as S

        cobertura, exceso = [], []
        for i in range(12):
            par = S.generar_par(7 * 10_000 + 1_000 + i, 'Estructurado')
            r = V.bounded_region(par.con_texto, par.caja)
            mask = V.generate_mask(par.con_texto, r, self.opts)
            ref = par.referencia[r.y : r.y + r.h, r.x : r.x + r.w]
            cobertura.append(M.cobertura(mask, ref))
            exceso.append(M.exceso(mask, ref))
        self.assertGreater(np.mean(cobertura), 0.7)
        self.assertLess(np.mean(exceso), 0.35)

    @staticmethod
    def texto_con_contorno(fondo):
        """Texto negro con borde blanco de 3 px sobre un fondo gris parejo."""
        img = np.full((140, 260, 3), fondo, np.uint8)
        cv2.putText(img, 'HOLA', (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,) * 3, 10)
        cv2.putText(img, 'HOLA', (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        r = V.bounded_region(img, (0, 0, 260, 140))
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), V.generate_mask(
            img, r, V.Options()
        )

    def test_incluye_el_contorno_blanco_sobre_trama_gris(self):
        # Sobre una trama gris media el borde blanco también hay que borrarlo.
        gris, mask = self.texto_con_contorno(170)
        self.assertGreater(float((mask[gris > 240] > 0).mean()), 0.9)
        self.assertGreater(float((mask[gris < 20] > 0).mean()), 0.9)
        self.assertEqual(int((mask[gris == 170] > 0).sum()), 0)

    def test_texto_con_contorno_sobre_fondo_oscuro_parejo(self):
        # Era una limitación: el fondo oscuro tocaba el borde de la selección y se
        # tomaba por una línea larga. Ahora se marcan las letras y casi nada del fondo.
        gris, mask = self.texto_con_contorno(45)
        self.assertGreater(float((mask[gris < 20] > 0).mean()), 0.9)
        self.assertLess(float((mask[gris == 45] > 0).mean()), 0.05)

    @unittest.expectedFailure
    def test_limitacion_letras_blancas_sobre_fondo_negro(self):
        # Limitación conocida: la máscara busca tinta oscura, así que no encuentra
        # letras claras sobre un fondo negro.
        img = np.full((140, 260, 3), 20, np.uint8)
        cv2.putText(img, 'HOLA', (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,) * 3, 5)
        r = V.bounded_region(img, (15, 30, 245, 115))
        mask = V.generate_mask(img, r, V.Options())
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[r.y : r.y + r.h, r.x : r.x + r.w]
        self.assertGreater(float((mask[gris > 200] > 0).mean()), 0.9)

    def test_sin_contorno_en_globo_blanco(self):
        # En un globo blanco no hay contorno que agregar: la máscara no cubre papel.
        r = V.bounded_region(self.img, TEXTO_LISO)
        mask = V.generate_mask(self.img, r, self.opts)
        gris = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)[
            r.y : r.y + r.h, r.x : r.x + r.w
        ]
        self.assertLess(float((gris[mask > 0] > 235).mean()), 0.1)

    def test_region_sin_tinta(self):
        img = np.full((100, 100, 3), 250, np.uint8)
        r = V.bounded_region(img, (10, 10, 90, 90))
        out, metodo, n = V.clean_region(img, r, self.opts)
        self.assertEqual(n, 0)
        self.assertEqual(metodo, 'Sin trazos detectados')
        self.assertTrue(np.array_equal(out, img))


class FondoTest(unittest.TestCase):
    def setUp(self):
        self.img = V.bgr_from_file(str(DEMO))
        self.opts = V.Options()

    def clasificar(self, caja):
        r = V.bounded_region(self.img, caja)
        V.generate_mask(self.img, r, self.opts)
        return r

    def test_globo_blanco_es_uniforme(self):
        r = self.clasificar(TEXTO_LISO)
        self.assertEqual(r.classification, 'Uniforme')
        self.assertEqual(r.suggested_method, 'Relleno local')

    def test_globo_con_trama_es_estructurado(self):
        r = self.clasificar(TEXTO_TRAMA)
        self.assertEqual(r.classification, 'Estructurado')
        self.assertEqual(r.suggested_method, 'Inpainting Telea')

    def test_una_vineta_entera_queda_para_revision(self):
        # Una viñeta completa (como los globos grandes que propone detect_bubbles)
        # no se limpia sola: su máscara incluiría dibujo.
        r = self.clasificar((720, 500, 1390, 970))
        self.assertEqual(r.classification, 'Revisar máscara')
        out, _, n = V.clean_region(self.img, r, self.opts)
        self.assertEqual(n, 0)
        self.assertTrue(np.array_equal(out, self.img))

    def test_en_una_imagen_recortada_el_tamano_no_bloquea(self):
        # En una imagen chica cualquier selección es gran parte de ella; bajo
        # AREA_REVISAR píxeles el tamaño no la deja para revisión.
        recorte = self.img[300:480, 380:640].copy()  # alrededor de «¿DÓNDE ESTÁS?»
        r = V.bounded_region(recorte, (35, 40, 210, 145))
        V.generate_mask(recorte, r, self.opts)
        self.assertEqual(r.classification, 'Uniforme')


class LimpiezaTest(unittest.TestCase):
    """La limpieza nunca debe modificar píxeles fuera de la máscara (meta del Hito 1)."""

    def setUp(self):
        self.img = V.bgr_from_file(str(DEMO))

    def comprobar(self, caja, metodo):
        r = V.bounded_region(self.img, caja)
        opts = V.Options(method=metodo)
        V.generate_mask(self.img, r, opts)
        out, usado, n = V.clean_region(self.img, r, opts)
        self.assertGreater(n, 0)
        cambiados = np.any(out != self.img, axis=2)
        fuera = cambiados & (mascara_completa(self.img, r) == 0)
        self.assertEqual(
            int(fuera.sum()), 0, f'{metodo} cambió píxeles fuera de la máscara'
        )
        self.assertGreater(int(cambiados.sum()), 0, f'{metodo} no cambió nada')
        return out, usado, r

    def test_relleno_local(self):
        out, usado, r = self.comprobar(TEXTO_LISO, 'Relleno local')
        self.assertEqual(usado, 'Relleno local')
        # En un globo blanco, lo limpiado debe quedar blanco.
        zona = out[r.y : r.y + r.h, r.x : r.x + r.w][r.mask > 0]
        self.assertGreater(float(zona.mean()), 240)

    def test_inpainting_telea(self):
        self.comprobar(TEXTO_TRAMA, 'Inpainting Telea')

    def test_inpainting_navier_stokes(self):
        self.comprobar(TEXTO_PUNTOS, 'Inpainting NS')

    def test_automatico_usa_el_metodo_sugerido(self):
        _, usado, r = self.comprobar(TEXTO_TRAMA, 'Automático')
        self.assertEqual(usado, r.suggested_method)

    def test_revision_manual_no_modifica_la_imagen(self):
        r = V.bounded_region(self.img, TEXTO_LISO)
        V.generate_mask(self.img, r, V.Options())
        out, usado, n = V.clean_region(self.img, r, V.Options(method='Revisión manual'))
        self.assertEqual(n, 0)
        self.assertTrue(np.array_equal(out, self.img))


def interseccion(a, b):
    """Área común de dos rectángulos (x1, y1, x2, y2)."""
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def area(c):
    return (c[2] - c[0]) * (c[3] - c[1])


class ZonaTest(unittest.TestCase):
    """texto_en_zona: de un rectángulo holgado a regiones ajustadas al texto."""

    def test_ajusta_cada_globo_del_ejemplo(self):
        img = V.bgr_from_file(str(DEMO))
        for texto in (TEXTO_LISO, TEXTO_PUNTOS, TEXTO_TRAMA):
            # Zona holgada: 40 px más por lado, con dibujo, trama y borde del globo.
            zona = (texto[0] - 40, texto[1] - 40, texto[2] + 40, texto[3] + 40)
            bloques = V.texto_en_zona(img, zona)
            self.assertEqual(len(bloques), 1, f'zona {zona}')
            b = bloques[0]
            caja = (b.x, b.y, b.x + b.w, b.y + b.h)
            self.assertGreater(interseccion(caja, texto), 0.85 * area(texto))
            self.assertLess(area(caja), 0.6 * area(zona), 'debe ajustarse al texto')

    def test_globo_sobre_fondo_negro_y_sin_pelos_del_dibujo(self):
        # Globo blanco sobre fondo negro (desvía el umbral de Otsu de la zona) y,
        # al lado, una zona blanca con pelos finos y grises del dibujo.
        img = np.zeros((300, 420, 3), np.uint8)
        cv2.ellipse(img, (110, 150), (90, 120), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (110, 150), (90, 120), 0, 0, 360, (0, 0, 0), 3)
        for i, letra in enumerate('ABC'):
            cv2.putText(
                img,
                letra,
                (90, 115 + 40 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 0),
                3,
            )
        img[40:260, 240:400] = 255
        for i in range(6):
            cv2.line(img, (250 + 20 * i, 60), (270 + 20 * i, 240), (150, 150, 150), 1)
        bloques = V.texto_en_zona(img, (5, 5, 415, 295))
        self.assertEqual(len(bloques), 1)
        b = bloques[0]
        self.assertTrue(70 <= b.x and b.x + b.w <= 150, (b.x, b.w))
        self.assertTrue(70 <= b.y and b.y + b.h <= 215, (b.y, b.h))

    def test_sin_texto_no_devuelve_nada(self):
        img = np.full((200, 200, 3), 255, np.uint8)
        cv2.circle(img, (100, 100), 60, (0, 0, 0), 2)
        self.assertEqual(V.texto_en_zona(img, (10, 10, 190, 190)), [])


class DetectorTextoTest(unittest.TestCase):
    """Detector entrenado (candidatos + random forest) y puntos del usuario."""

    def setUp(self):
        self.img = V.bgr_from_file(str(DEMO))

    def caja_parecida(self, regiones, texto):
        """La región que cubre la mayor parte de la caja de texto dada, o None."""
        for r in regiones:
            caja = (r.x, r.y, r.x + r.w, r.y + r.h)
            if interseccion(caja, texto) > 0.8 * area(texto):
                return r
        return None

    def test_detecta_el_globo_blanco_con_confianza(self):
        r = self.caja_parecida(V.detectar_texto(self.img), TEXTO_LISO)
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r.confianza, V.CONFIANZA_LIMPIAR)
        self.assertEqual(r.origin, 'Detectada')

    def test_un_punto_confirma_el_texto_que_tiene_cerca(self):
        x1, y1, x2, y2 = TEXTO_PUNTOS
        punto = ((x1 + x2) // 2, (y1 + y2) // 2)
        r = self.caja_parecida(V.detectar_texto(self.img, [punto]), TEXTO_PUNTOS)
        self.assertIsNotNone(r)
        self.assertEqual(r.confianza, 1.0)

    def test_un_clic_en_el_papel_del_globo_cuenta(self):
        papel = V.papel_de_pagina(self.img)
        r = self.caja_parecida(
            V.puntuar_candidatos(self.img, papel=papel), TEXTO_PUNTOS
        )
        arriba_del_texto = (1190, 140)  # papel del globo, a 28 px de las letras
        self.assertEqual(V.regiones_del_punto([r], arriba_del_texto, papel), [r])
        self.assertEqual(
            V.regiones_del_punto([r], (700, 900), papel), [], 'otra viñeta'
        )

    def test_encuentra_el_texto_con_papel_gris_tinta_lavada_o_mas_resolucion(self):
        f = self.img.astype(np.float32)
        variantes = {
            'papel gris': np.clip(f * 0.8, 0, 255),
            'papel amarillento': np.clip(f * np.array([0.72, 0.86, 0.92]), 0, 255),
            'tinta lavada': np.clip(f * 0.62 + 80, 0, 255),
        }
        for nombre, imagen in variantes.items():
            r = self.caja_parecida(
                V.detectar_texto(imagen.astype(np.uint8)), TEXTO_LISO
            )
            self.assertIsNotNone(r, nombre)
            self.assertGreaterEqual(r.confianza, V.CONFIANZA_LIMPIAR, nombre)
        doble = cv2.resize(self.img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        texto = tuple(2 * v for v in TEXTO_LISO)
        self.assertIsNotNone(self.caja_parecida(V.detectar_texto(doble), texto))

    def test_una_hoja_casi_en_blanco_no_oscurece_sus_grises(self):
        gris = np.full((1170, 830), 255, np.uint8)
        gris[500:1000, 100:700] = 185  # trama clara
        for k, letra in enumerate('HOLA'):
            cv2.putText(
                gris, letra, (200 + 24 * k, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 3
            )
        self.assertIs(V.aclarar_papel(gris), gris)

    def test_una_pagina_con_buen_contraste_no_se_altera(self):
        gris = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.assertIs(V.aclarar_papel(gris), gris)
        self.assertEqual(V.preparar_pagina(self.img).escala, 1.0)

    def test_figuras_grandes_no_se_toman_por_letras(self):
        # Papel con cuatro figuras grandes: antes, la hoja entera era un «bloque».
        img = np.full((1170, 830, 3), 255, np.uint8)
        for x, y in ((80, 80), (460, 80), (80, 620), (460, 620)):
            cv2.rectangle(img, (x, y), (x + 280, y + 400), (0, 0, 0), -1)
        self.assertEqual(V.texto_en_zona(img, (0, 0, 830, 1170)), [])

    def test_una_palabra_en_letra_latina_es_un_bloque(self):
        img = np.full((400, 400, 3), 255, np.uint8)
        # Letras que se tocan: la palabra entera es una sola pieza muy alargada, que
        # antes se descartaba como si fuera una línea del dibujo.
        cv2.putText(
            img, 'HELLO', (120, 210), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 3
        )
        cv2.line(img, (122, 209), (225, 209), (0, 0, 0), 2)
        self.assertEqual(len(V.texto_en_zona(img, (20, 20, 380, 380))), 1)
        # Una línea del dibujo, del mismo largo pero delgada, se sigue descartando.
        linea = np.full((400, 400, 3), 255, np.uint8)
        cv2.line(linea, (120, 200), (230, 200), (0, 0, 0), 4)
        self.assertEqual(V.texto_en_zona(linea, (20, 20, 380, 380)), [])

    def test_un_punto_no_confirma_una_region_gigante(self):
        papel = V.papel_de_pagina(self.img)
        alto, ancho = self.img.shape[:2]
        gigante = V.Region(0, 0, ancho, alto, confianza=0.35)
        chica = V.Region(1097, 168, 190, 86, confianza=0.35)
        punto = (1190, 210)
        self.assertEqual(V.regiones_del_punto([gigante], punto, papel), [])
        self.assertEqual(V.regiones_del_punto([gigante, chica], punto, papel), [chica])
        elegidas = V.elegir_texto([gigante, chica], (punto,), papel)
        self.assertEqual([r.confianza for r in elegidas], [0.35, 1.0])

    def test_un_punto_encuentra_texto_en_un_globo_gris(self):
        # Globo gris sobre papel blanco: con el papel de la página no hay candidato.
        img = np.full((600, 600, 3), 255, np.uint8)
        cv2.ellipse(img, (300, 300), (150, 110), 0, 0, 360, (150, 150, 150), -1)
        cv2.ellipse(img, (300, 300), (150, 110), 0, 0, 360, (0, 0, 0), 3)
        for k, letra in enumerate('ABCD'):
            cv2.putText(
                img,
                letra,
                (215 + 45 * k, 315),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                4,
            )
        regiones = V.buscar_en_punto(img, (300, 260))
        self.assertEqual(len(regiones), 1)
        r = regiones[0]
        self.assertTrue(190 <= r.x and r.x + r.w <= 420 and 260 <= r.y, r.rect)

    def test_sin_punto_cerca_no_se_pierde_nada(self):
        sin = V.detectar_texto(self.img)
        lejos = V.detectar_texto(self.img, [(5, 5)])
        self.assertEqual([r.rect for r in sin], [r.rect for r in lejos])
        self.assertEqual([r.confianza for r in sin], [r.confianza for r in lejos])

    def test_una_region_dudosa_queda_para_revision(self):
        r = V.bounded_region(self.img, TEXTO_LISO)
        r.confianza = V.CONFIANZA_LIMPIAR - 0.1
        V.generate_mask(self.img, r, V.Options())
        self.assertEqual(r.suggested_method, 'Revisión manual')
        self.assertIn('Puede no ser texto', r.review_reason)
        sin_cambios, _, n = V.clean_region(self.img, r, V.Options())
        self.assertEqual(n, 0)
        self.assertTrue(np.array_equal(sin_cambios, self.img))

    def test_sin_modelo_propone_pero_no_limpia(self):
        with mock.patch.object(V, 'cargar_modelo', return_value=None):
            regiones = V.puntuar_candidatos(self.img)
        self.assertTrue(regiones)
        self.assertTrue(all(r.confianza == V.CONFIANZA_PROPONER for r in regiones))

    def test_el_modelo_se_lee_aunque_la_ruta_tenga_tildes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / 'Tomás Ñandú' / 'modelo_texto.yml'
            ruta.parent.mkdir()
            ruta.write_bytes(V.MODELO_TEXTO.read_bytes())
            modelo = V.cargar_modelo(ruta)
            V._modelos.pop(ruta, None)
        _, medidas = V.candidatos_de_texto(self.img)
        self.assertTrue(
            np.allclose(
                V.probabilidad_de_texto(modelo, medidas),
                V.probabilidad_de_texto(V.cargar_modelo(), medidas),
            )
        )


class DeteccionTest(unittest.TestCase):
    def test_encuentra_los_tres_globos_del_ejemplo(self):
        img = V.bgr_from_file(str(DEMO))
        globos = V.detect_bubbles(img)
        for x1, y1, x2, y2 in (TEXTO_LISO, TEXTO_PUNTOS, TEXTO_TRAMA):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.assertTrue(
                any(g.contains(cx, cy) for g in globos),
                f'falta el globo en ({cx}, {cy})',
            )


if __name__ == '__main__':
    unittest.main()
