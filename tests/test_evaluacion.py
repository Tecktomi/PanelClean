"""Pruebas de la evaluación: métricas (evaluacion/metricas.py) y pares sintéticos.

Ejecutar desde la raíz del repositorio:
    python -m unittest discover tests
"""

import math
import shutil
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from evaluacion import cajas as K
from evaluacion import calibrar_selector as C
from evaluacion import entrenar_detector as D
from evaluacion import horizontales, marcar
from evaluacion import evaluar as E
from evaluacion import metricas as M
from evaluacion import muestreo
from evaluacion import sinteticos as S
from panelclean import vision as V

try:
    import pypdfium2  # noqa: F401
except ImportError:
    pypdfium2 = None

DEMO = Path(V.__file__).parent.parent / 'examples' / 'demo_pagina.png'


class MascaraTest(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((10, 10), np.uint8)
        self.a[:, :4] = 255  # 40 píxeles
        self.b = np.zeros((10, 10), np.uint8)
        self.b[:, 2:6] = 255  # 40 píxeles, 20 en común con a

    def test_mascaras_iguales_y_disjuntas(self):
        self.assertEqual(M.iou(self.a, self.a), 1.0)
        self.assertEqual(M.dice(self.a, self.a), 1.0)
        self.assertEqual(M.iou(self.a, 255 - self.a), 0.0)

    def test_valores_conocidos(self):
        self.assertAlmostEqual(M.iou(self.a, self.b), 20 / 60)
        self.assertAlmostEqual(M.dice(self.a, self.b), 40 / 80)
        self.assertAlmostEqual(M.cobertura(self.a, self.b), 0.5)
        self.assertAlmostEqual(M.exceso(self.a, self.b), 0.5)

    def test_mascaras_vacias(self):
        vacia = np.zeros((5, 5), np.uint8)
        self.assertEqual(M.iou(vacia, vacia), 1.0)
        self.assertEqual(M.exceso(vacia, self.a[:5, :5]), 0.0)


class LimpiezaTest(unittest.TestCase):
    def test_mae_y_psnr(self):
        a = np.full((20, 20), 100, np.uint8)
        b = np.full((20, 20), 110, np.uint8)
        self.assertAlmostEqual(M.mae(a, b), 10.0)
        self.assertAlmostEqual(M.psnr(a, b), 10 * math.log10(255**2 / 100), places=6)
        self.assertEqual(M.psnr(a, a), 60.0)

    def test_mae_en_zona(self):
        a = np.zeros((4, 4), np.uint8)
        b = a.copy()
        b[0, 0] = 40
        zona = np.zeros((4, 4), np.uint8)
        zona[0, :2] = 1
        self.assertAlmostEqual(M.mae(a, b, zona), 20.0)

    def test_ssim(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (64, 64)).astype(np.uint8)
        self.assertAlmostEqual(M.ssim(img, img), 1.0, places=9)
        ruido = np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
        # Con varianza local ≈ 5461 y ruido de varianza 400, SSIM ≈ 2·5461/(2·5461+400) ≈ 0,965.
        self.assertAlmostEqual(M.ssim(img, ruido), 0.965, delta=0.01)
        self.assertAlmostEqual(M.ssim(img, ruido), M.ssim(ruido, img), places=9)

    def test_ssim_de_imagenes_constantes(self):
        # Sin varianza, SSIM = (2ab + C1) / (a² + b² + C1).
        a = np.full((32, 32), 100, np.uint8)
        b = np.full((32, 32), 110, np.uint8)
        c1 = (0.01 * 255) ** 2
        esperado = (2 * 100 * 110 + c1) / (100**2 + 110**2 + c1)
        self.assertAlmostEqual(M.ssim(a, b), esperado, places=6)


class DeteccionTest(unittest.TestCase):
    def test_cobertura(self):
        textos = [(10, 10, 20, 40), (100, 100, 110, 130)]
        detecciones = [
            (0, 0, 40, 50),
            (300, 300, 350, 350),
        ]  # un globo y un falso positivo
        acertadas, correctas = M.aciertos_por_cobertura(detecciones, textos)
        self.assertEqual((acertadas, correctas), (1, 1))
        p, r, f1 = M.precision_recall_f1(correctas, 2, acertadas, 2)
        self.assertEqual((p, r, f1), (0.5, 0.5, 0.5))

    def test_iou_uno_a_uno(self):
        textos = [(0, 0, 10, 10)]
        detecciones = [
            (0, 0, 10, 10),
            (0, 0, 10, 11),
        ]  # dos detecciones del mismo texto
        self.assertEqual(M.aciertos_por_iou(detecciones, textos), 1)
        self.assertEqual(M.aciertos_por_iou([(0, 0, 10, 30)], textos), 0)  # IoU 1/3


class SelectorTest(unittest.TestCase):
    def test_macro_f1(self):
        reales = ['A', 'A', 'B', 'B']
        predichas = ['A', 'B', 'B', 'B']
        self.assertAlmostEqual(M.exactitud(reales, predichas), 0.75)
        # F1(A) = 2/3, F1(B) = 0,8.
        self.assertAlmostEqual(
            M.macro_f1(reales, predichas, ['A', 'B']), (2 / 3 + 0.8) / 2
        )
        np.testing.assert_array_equal(
            M.matriz_confusion(reales, predichas, ['A', 'B']), [[1, 1], [0, 2]]
        )

    def test_macro_f1_ignora_clases_ausentes(self):
        # «C» no está ni en las reales ni en las predichas: no baja el promedio.
        self.assertEqual(M.macro_f1(['A', 'B'], ['A', 'B'], ['A', 'B', 'C']), 1.0)

    def test_metricas_ponderadas(self):
        reales, predichas = ['A', 'B'], ['A', 'A']
        # La región B (mal clasificada) pesa 3: la exactitud ponderada es 1/4.
        self.assertAlmostEqual(M.exactitud(reales, predichas, [1, 3]), 0.25)
        np.testing.assert_array_equal(
            M.matriz_confusion(reales, predichas, ['A', 'B'], [1, 3]), [[1, 0], [3, 0]]
        )
        # Pesos iguales dan lo mismo que sin pesos.
        self.assertAlmostEqual(
            M.macro_f1(reales, predichas, ['A', 'B'], [2, 2]),
            M.macro_f1(reales, predichas, ['A', 'B']),
        )

    def test_kappa(self):
        self.assertEqual(M.kappa(['a', 'b', 'a', 'b'], ['a', 'b', 'a', 'b']), 1.0)
        self.assertAlmostEqual(M.kappa(['a', 'a', 'b', 'b'], ['a', 'b', 'a', 'b']), 0.0)
        # Coinciden 9 de 10, pero solo porque casi todo es de una clase.
        self.assertAlmostEqual(M.kappa(['a'] * 10, ['a'] * 9 + ['b']), 0.0)

    def test_dos_pasadas_de_una_persona_no_son_acuerdo_entre_personas(self):
        filas = [
            'libro,pagina,texto,etiqueta,etiquetador',
            'x,0,0,Uniforme,tomas',
            'x,0,0,Uniforme,tomas ramdohr',
            'x,0,1,Estructurado,tomas',
            'x,0,1,Uniforme,tomas ramdohr',
            'x,0,2,Complejo,tomas',
            'x,0,2,Complejo,ana',
        ]
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'etiquetas.csv'
            ruta.write_text('\n'.join(filas), encoding='utf8')
            with mock.patch.object(E, 'ETIQUETAS', ruta):
                consenso, acuerdo = E.leer_etiquetas()
        self.assertEqual([c['texto'] for c in consenso], [0, 2])
        self.assertEqual(acuerdo['personas'], 2)
        self.assertEqual(len(acuerdo['misma']), 2)
        self.assertEqual(acuerdo['entre'], [('Complejo', 'Complejo')])
        texto = ' '.join(E.resumen_coincidencia(acuerdo))
        self.assertIn('no acuerdo entre personas', texto)

    def test_regla_del_selector(self):
        # (entropía en bits, tono oscuro): papel, trama gris y dibujo con tinta.
        self.assertEqual(V.classify_background(0.3, 250)[0], 'Uniforme')
        self.assertEqual(V.classify_background(2.8, 150)[0], 'Estructurado')
        self.assertEqual(V.classify_background(3.5, 5)[0], 'Complejo / revisar')
        self.assertEqual(V.classify_background(0.3, 250, 0.5)[0], 'Revisar máscara')
        # Región grande: queda para revisión solo si es grande en píxeles y en la página.
        self.assertEqual(
            V.classify_background(0.3, 250, area=90_000, page_fraction=0.1)[0],
            'Revisar máscara',
        )
        self.assertEqual(
            V.classify_background(0.3, 250, area=90_000, page_fraction=0.03)[0],
            'Uniforme',
        )
        self.assertEqual(
            V.classify_background(0.3, 250, area=30_000, page_fraction=0.4)[0],
            'Uniforme',
        )
        # Trama rala: pocos tonos, pero repartida por toda la zona.
        self.assertEqual(
            V.classify_background(0.8, 170, texture=0.97)[0], 'Estructurado'
        )
        # La calibración puede probar otros umbrales sin tocar los de la aplicación.
        self.assertEqual(
            V.classify_background(2.8, 150, umbrales=(3.0, 40))[0], 'Uniforme'
        )

    def test_entropia(self):
        self.assertEqual(V._entropia(np.full(100, 250, np.uint8)), 0.0)
        # Los 256 tonos por igual llenan los 32 niveles: log2(32) = 5 bits.
        self.assertAlmostEqual(V._entropia(np.arange(256, dtype=np.uint8)), 5.0)

    def test_textura_repartida(self):
        papel = np.full((60, 60), 245, np.uint8)
        zona = np.ones(papel.shape, bool)
        self.assertEqual(V._textura_repartida(papel, zona), 0.0)
        # Una sola línea: solo varía cerca de ella.
        linea = papel.copy()
        linea[30, :] = 60
        self.assertLess(V._textura_repartida(linea, zona), 0.3)
        # Rayado fino cada 5 px: varía en toda la zona.
        rayado = papel.copy()
        rayado[::5, :] = 150
        self.assertGreater(V._textura_repartida(rayado, zona), 0.9)

    def test_calibracion_encuentra_umbrales_que_separan(self):
        def filas(libro, n=4):
            return [
                {
                    'libro': libro,
                    'clase': c,
                    'peso': 1.0,
                    'entropia': e,
                    'tono_oscuro': t,
                    'textura': 0.0,
                }
                for c, e, t in (
                    ('Uniforme', 0.4, 245),
                    ('Estructurado', 2.6, 130),
                    ('Complejo', 3.6, 8),
                )
                for _ in range(n)
            ]

        reales = filas('a') + filas('b') + filas('c')
        umbrales = C.ajustar(filas('sintético'), reales)
        self.assertEqual(C.macro_f1(reales, umbrales), 1.0)
        self.assertTrue(0.4 < umbrales[0] <= 2.6 and 8 < umbrales[1] <= 130)
        cv = C.validacion_cruzada(filas('sintético'), reales)
        self.assertEqual(cv['macro_f1'], 1.0)
        self.assertEqual(set(cv['umbrales']), {'a', 'b', 'c'})

    def test_la_obra_de_prueba_nunca_se_usa_para_ajustar(self):
        reales = [
            {
                'libro': libro,
                'clase': c,
                'peso': 1.0,
                'entropia': e,
                'tono_oscuro': t,
                'textura': 0.0,
            }
            for libro in ('a', 'b', 'prueba')
            for c, e, t in (('Uniforme', 0.4, 245), ('Complejo', 3.6, 8))
        ]
        with mock.patch.object(C, 'ajustar', wraps=C.ajustar) as ajustar:
            cv = C.validacion_cruzada(reales[:2], reales, reservada='prueba')
        usadas = {
            f['libro'] for llamada in ajustar.call_args_list for f in llamada.args[1]
        }
        self.assertNotIn('prueba', usadas)
        self.assertIn(
            'prueba', cv['umbrales']
        )  # se predice, pero no se ajusta con ella

    def test_bootstrap(self):
        media, inf, sup = M.intervalo_bootstrap([3.0] * 10)
        self.assertEqual((media, inf, sup), (3.0, 3.0, 3.0))
        media, inf, sup = M.intervalo_bootstrap(list(range(100)))
        self.assertLess(inf, media)
        self.assertLess(media, sup)


class SinteticosTest(unittest.TestCase):
    def test_misma_semilla_mismo_par(self):
        a = S.generar_par(123, 'Complejo')
        b = S.generar_par(123, 'Complejo')
        np.testing.assert_array_equal(a.con_texto, b.con_texto)
        np.testing.assert_array_equal(a.referencia, b.referencia)

    def test_referencia_coherente(self):
        for clase in S.CLASES:
            par = S.generar_par(7, clase)
            self.assertEqual(par.clase, clase)
            self.assertGreater(int((par.referencia > 0).sum()), 200)
            # Fuera de la referencia el texto no cambió el fondo más que el umbral.
            dif = np.abs(par.con_texto.astype(int) - par.limpio.astype(int)).max(axis=2)
            self.assertLessEqual(
                int(dif[par.referencia == 0].max()), S.UMBRAL_REFERENCIA
            )
            # Todo el texto queda dentro de la selección, con margen.
            x1, y1, x2, y2 = par.caja
            fuera = par.referencia.copy()
            fuera[y1:y2, x1:x2] = 0
            self.assertEqual(int(fuera.sum()), 0)


class EvaluacionTest(unittest.TestCase):
    def test_evaluar_sinteticos(self):
        pares = S.generar_conjunto(por_clase=1)
        mascara, limpieza = E.evaluar_sinteticos(pares)
        self.assertEqual(len(mascara), 3)
        # Por par: «sin limpiar» + 2 máscaras × 4 métodos.
        self.assertEqual(len(limpieza), 3 * (1 + 2 * len(E.METODOS)))
        self.assertEqual(max(f['px_fuera_mascara'] for f in limpieza), 0)
        for f in mascara:
            self.assertTrue(0 <= f['iou'] <= 1)
            self.assertIn(f['selector'], S.CLASES)
        texto, datos = E.resumen_limpieza(limpieza)
        self.assertIn('Hipótesis', texto)
        self.assertEqual(set(datos['perfecta']), {'MAE', 'PSNR', 'SSIM'})
        # Meta de limpieza: alguna métrica mejora y ninguna empeora.
        igual = 'sin diferencia significativa'
        for mae, psnr, cumple in (
            (igual, 'mejor', True),
            ('peor', 'mejor', False),
            (igual, igual, False),
        ):
            veredictos = {'MAE': mae, 'PSNR': psnr, 'SSIM': igual}
            self.assertEqual(E.meta_limpieza(veredictos), cumple, veredictos)


class MuestreoTest(unittest.TestCase):
    def setUp(self):
        self.estratos = muestreo.cargar_estratos()

    def test_cuatro_estratos_con_las_proporciones_esperadas(self):
        tamanos = {h: list(self.estratos.values()).count(h) for h in (1, 2, 3, 4)}
        total = sum(tamanos.values())
        self.assertEqual(total, 1592)
        self.assertAlmostEqual(tamanos[1] / total, 0.05, delta=0.01)
        self.assertAlmostEqual(tamanos[4] / total, 0.50, delta=0.01)

    def test_orden_completo_y_cargado_al_principio(self):
        claves = list(self.estratos)
        orden = muestreo.ordenar(claves)
        self.assertEqual(sorted(orden), sorted(claves))
        self.assertEqual(orden, muestreo.ordenar(claves))  # siempre el mismo
        primeras = [self.estratos[c] for c in orden[:40]]
        self.assertGreater(primeras.count(1), 15)

    def test_pesos(self):
        una_por_estrato = []
        for h in (1, 2, 3, 4):
            una_por_estrato.append(next(c for c, e in self.estratos.items() if e == h))
        pesos, por_estrato = muestreo.pesos(una_por_estrato)
        for c in una_por_estrato:
            n, total = por_estrato[self.estratos[c]]
            self.assertEqual(n, 1)
            self.assertEqual(pesos[c], total)


class DetectorTest(unittest.TestCase):
    """Entrenamiento del detector de texto (entrenar_detector.py)."""

    def fila(self, grupo, valor, es_texto):
        return {
            'grupo': grupo,
            'es_texto': str(es_texto),
            'enteros': '0' if es_texto else '',
            **{r: str(valor) for r in V.RASGOS_TEXTO},
        }

    def test_el_modelo_entrenado_se_puede_guardar_y_leer(self):
        X = np.random.default_rng(0).random((200, len(V.RASGOS_TEXTO)), np.float32)
        y = (X[:, 0] > 0.5).astype(np.int32)
        modelo = D.entrenar(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / 'modelo.yml'
            D.guardar(modelo, ruta)
            leido = V.cargar_modelo(ruta)
            V._modelos.pop(ruta, None)
        p = V.probabilidad_de_texto(leido, X)
        self.assertTrue(np.allclose(p, V.probabilidad_de_texto(modelo, X)))
        self.assertGreater(((p >= 0.5) == (y == 1)).mean(), 0.9)

    def test_lo_reservado_nunca_se_usa_para_entrenar(self):
        # Cada grupo tiene medidas propias; lo reservado, de 100 en adelante.
        filas = [
            self.fila(g, b + k, k % 2)
            for g, b in {'a': 0, 'b': 10, D.LIBRO_PRUEBA: 100}.items()
            for k in range(6)
        ]
        propias = [
            self.fila(g, b + k, k % 2)
            for g, b in {'mi_manga': 20, 'serie_reservada': 200}.items()
            for k in range(6)
        ]
        vistos = []
        original = D.entrenar

        def espia(X, y):
            vistos.append(set(X[:, 0].tolist()))
            return original(X, y)

        with mock.patch.object(D, 'entrenar', espia):
            p, pp, _ = D.validacion_cruzada(filas, propias, {'serie_reservada'})
        self.assertEqual((len(p), len(pp)), (len(filas), len(propias)))
        self.assertTrue(all(max(v) < 100 for v in vistos))
        # Final, sin «a», sin «b» y sin «mi_manga».
        self.assertEqual(len(vistos), 4)
        self.assertTrue(any(not v & {20.0, 21.0} for v in vistos), 'sin mi_manga')

    def test_las_paginas_propias_pesan_mas_que_una_vez(self):
        filas = [self.fila('a', k, k % 2) for k in range(100)]
        propias = [self.fila('mi_manga', 500 + k, k % 2) for k in range(4)]
        with mock.patch.object(D, 'entrenar') as entrenar:
            D.entrenar_con(filas, propias)
        X, y = entrenar.call_args.args
        self.assertEqual(int((X[:, 0] >= 500).sum()), 40)  # 4 ejemplos × 10
        self.assertEqual(len(X), len(y))

    def test_un_trozo_de_texto_no_es_ejemplo_de_nada(self):
        entero, trozo, dibujo = (self.fila('a', v, int(v < 3)) for v in (1, 2, 3))
        trozo['enteros'] = ''
        X, y = D.ejemplos([entero, trozo, dibujo])
        self.assertEqual(X[:, 0].tolist(), [1.0, 3.0])
        self.assertEqual(y.tolist(), [1, 0])

    def test_el_entrenamiento_sale_siempre_igual(self):
        X = np.random.default_rng(1).random((300, len(V.RASGOS_TEXTO)), np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0.7).astype(np.int32)
        a = V.probabilidad_de_texto(D.entrenar(X, y), X)
        cv2.setRNGSeed(99)  # lo que haya pasado antes no influye
        self.assertTrue(np.array_equal(a, V.probabilidad_de_texto(D.entrenar(X, y), X)))

    def test_se_mide_sin_contar_dos_veces_el_mismo_texto(self):
        def fila(x, y, w, h, texto, cubre):
            return {
                'grupo': 'a',
                'pagina': '1',
                'tipo': 'original',
                **dict(zip('xywh', map(str, (x, y, w, h)))),
                'es_texto': '1',
                'texto': str(texto),
                'cubre': cubre,
                'limpiable': '0',
            }

        filas = [
            fila(0, 0, 100, 90, 0, '0:1.00'),
            fila(5, 5, 90, 25, 0, '0:0.33'),  # un renglón del anterior
            fila(300, 0, 60, 20, 1, '1:0.40'),  # solo un trozo del segundo texto
        ]
        m = D.medir_deteccion(filas, np.array([0.8, 0.9, 0.9]), 0.5, 2)
        self.assertEqual(m['detecciones'], 2)
        self.assertEqual((m['precision'], m['recall']), (1.0, 1.0))
        self.assertEqual(m['enteros'], 0.5, 'el segundo texto queda a medias')

    def test_un_escaneo_chico_conserva_las_cajas(self):
        img = np.full((1200, 800, 3), 255, np.uint8)
        cv2.rectangle(img, (200, 300), (399, 499), (0, 0, 0), -1)
        chica, [caja] = horizontales.degradar(
            img, [(200, 300, 400, 500)], horizontales.azar('obra', 1)
        )
        self.assertTrue(500 <= chica.shape[1] < 620, chica.shape)
        x1, y1, x2, y2 = caja
        self.assertLess(chica[y1 + 3 : y2 - 3, x1 + 3 : x2 - 3].mean(), 10)
        self.assertGreater(chica[: y1 - 5].mean(), 245)

    def test_una_region_gigante_no_es_texto_aunque_su_centro_caiga_en_uno(self):
        texto = (100, 100, 150, 200)
        self.assertEqual(K.texto_del_candidato((95, 95, 155, 205), [texto]), 0)
        self.assertEqual(K.texto_del_candidato((0, 0, 250, 300), [texto]), -1)
        self.assertFalse(K.acierta((0, 0, 250, 300), texto))

    def test_un_candidato_que_cubre_dos_globos_unidos_es_texto(self):
        textos = [(0, 0, 50, 100), (60, 0, 110, 100)]
        self.assertEqual(K.texto_del_candidato((0, 0, 110, 100), textos), 0)

    def test_lo_que_se_ignora_no_cuenta_como_no_texto(self):
        img = np.full((400, 400, 3), 255, np.uint8)
        for k, letra in enumerate('ABC'):
            cv2.putText(
                img,
                letra,
                (60 + 40 * k, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                4,
            )
        comun = {'grupo': 'g', 'pagina': 'p', 'tipo': 'propia'}
        caja = (50, 80, 190, 135)
        marcada = D.medir_pagina(img, [caja], [], comun)
        self.assertEqual({f['es_texto'] for f in marcada}, {1})
        self.assertIn('0', [f['enteros'] for f in marcada])
        sin_marcar = D.medir_pagina(img, [], [], comun)
        self.assertEqual({f['es_texto'] for f in sin_marcar}, {0})
        self.assertEqual(D.medir_pagina(img, [], [caja], comun), [])

    def test_la_traduccion_simulada_es_reproducible(self):
        img = np.full((300, 300, 3), 255, np.uint8)
        textos = [{'x': 100, 'y': 60, 'w': 90, 'h': 160, 'text_en': 'hello there'}]
        a, ca = horizontales.traducir(img, textos, horizontales.azar('obra', 3))
        b, cb = horizontales.traducir(img, textos, horizontales.azar('obra', 3))
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(ca, cb)
        self.assertLess(a.min(), 100, 'debe haber texto escrito')


class MarcarTest(unittest.TestCase):
    """Programa para marcar texto (marcar.py), manejado por código."""

    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Tk no disponible: {exc}')
        self.root.withdraw()
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.carpeta = base / 'mi_manga'
        self.carpeta.mkdir()
        for nombre in ('001.png', '002.png'):
            shutil.copy(DEMO, self.carpeta / nombre)
        for nombre, valor in (
            ('CAJAS', base / 'cajas_texto.json'),
            ('LOCALES', base / 'carpetas_locales.json'),
        ):
            parche = mock.patch.object(K, nombre, valor)
            parche.start()
            self.addCleanup(parche.stop)

    def tearDown(self):
        self.root.destroy()
        self.tmp.cleanup()

    def abrir(self):
        ventana = marcar.Marcador(self.root, self.carpeta, 'Prueba')
        ventana.dibujar()
        return ventana

    def arrastrar(self, ventana, a, b):
        """Arrastre con el botón izquierdo entre dos puntos de la página."""
        escala, ox, oy = ventana._vista

        def evento(p):
            return mock.Mock(x=ox + p[0] * escala, y=oy + p[1] * escala)

        ventana.clic(evento(a))
        ventana.arrastre(evento(b))
        ventana.soltar(evento(b))

    def test_propone_lo_que_encuentra_el_detector(self):
        ventana = self.abrir()
        self.assertTrue(ventana.cajas)
        self.assertEqual({c['origen'] for c in ventana.cajas}, {'propuesta'})

    def test_dibujar_borrar_deshacer_y_guardar(self):
        ventana = self.abrir()
        ventana.borrar_todas()
        self.arrastrar(ventana, (200, 200), (500, 420))
        self.assertEqual(len(ventana.cajas), 1)
        c = ventana.cajas[0]
        self.assertEqual((c['tipo'], c['origen']), ('texto', 'mano'))
        self.assertAlmostEqual(c['x'], 200, delta=15)
        self.assertAlmostEqual(c['w'], 300, delta=30)
        ventana.cambiar_tipo()
        self.assertEqual(ventana.cajas[0]['tipo'], 'ignorar')
        ventana.deshacer()
        self.assertEqual(ventana.cajas[0]['tipo'], 'texto')
        ventana.pagina_lista()
        self.assertEqual(ventana.indice, 1, 'pasa a la siguiente')
        guardada = K.leer()['mi_manga']['paginas']['001.png']
        self.assertTrue(guardada['completa'])
        self.assertEqual(len(K.rectangulos(guardada, 'texto')), 1)
        self.assertNotIn(str(self.carpeta), K.CAJAS.read_text(encoding='utf8'))
        ventana.cambiar_pagina(-1)
        self.assertEqual(len(ventana.cajas), 1, 'se vuelve a ver lo marcado')

    def test_mover_una_caja_no_la_saca_de_la_pagina(self):
        ventana = self.abrir()
        ventana.borrar_todas()
        self.arrastrar(ventana, (300, 300), (600, 600))
        self.arrastrar(ventana, (450, 450), (5000, 5000))
        c = ventana.cajas[0]
        alto, ancho = ventana.imagen.shape[:2]
        self.assertEqual((c['x'] + c['w'], c['y'] + c['h']), (ancho, alto))

    def test_una_serie_reservada_no_muestra_propuestas(self):
        ventana = self.abrir()
        ventana.reservada.set(True)
        ventana.cambiar_reservada()
        self.assertEqual(ventana.cajas, [])
        ventana.cambiar_pagina(1)
        self.assertEqual(ventana.cajas, [])
        ventana.pagina_lista()
        self.assertTrue(K.leer()['mi_manga']['reservada'])

    def test_una_pagina_sin_terminar_no_entra_al_entrenamiento(self):
        ventana = self.abrir()
        ventana.borrar_todas()
        self.arrastrar(ventana, (200, 200), (500, 420))
        ventana.cambiar_pagina(1)  # la primera queda como borrador
        ventana.pagina_lista()
        serie = K.leer()['mi_manga']
        self.assertEqual(set(serie['paginas']), {'001.png', '002.png'})
        self.assertEqual(set(K.completas(serie)), {'002.png'})
        with mock.patch.object(D, 'PROPIOS', Path(self.tmp.name) / 'propios.csv'):
            filas = D.medir_propias(avance=lambda _: None)
        self.assertEqual({f['pagina'] for f in filas}, {'002.png'})
        self.assertTrue(any(f['es_texto'] == '1' for f in filas))

    def test_sin_las_imagenes_se_usan_las_medidas_guardadas(self):
        ventana = self.abrir()
        ventana.pagina_lista()
        with mock.patch.object(D, 'PROPIOS', Path(self.tmp.name) / 'propios.csv'):
            antes = D.medir_propias(avance=lambda _: None)
            K.LOCALES.unlink()  # como en el equipo de otra persona
            despues = D.medir_propias(avance=lambda _: None)
            self.assertEqual(antes, despues)
            with mock.patch.object(V, 'VERSION_RASGOS', 999):
                self.assertEqual(D.medir_propias(avance=lambda _: None), [])

    def test_una_carpeta_sin_imagenes_no_falla(self):
        vacia = Path(self.tmp.name) / 'vacia'
        vacia.mkdir()
        ventana = marcar.Marcador(self.root, vacia, 'Prueba')
        ventana.reservada.set(True)
        ventana.cambiar_reservada()
        ventana.pagina_lista()
        ventana.cambiar_pagina(1)
        self.assertFalse(K.CAJAS.exists(), 'no guarda una serie vacía')

    def crear_pdf(self, destino, paginas=2):
        """PDF con la página de ejemplo repetida, como un manga escaneado."""
        imagen = Image.open(self.carpeta / '001.png').convert('RGB')
        imagen.save(destino, save_all=True, append_images=[imagen] * (paginas - 1))
        return destino

    @unittest.skipUnless(pypdfium2, 'pypdfium2 no está instalado')
    def test_una_carpeta_con_pdf_muestra_todas_sus_paginas(self):
        self.crear_pdf(self.carpeta / 'tomo 2.PDF')
        (self.carpeta / 'roto.pdf').write_bytes(b'nada')
        with mock.patch.object(marcar.messagebox, 'showwarning') as aviso:
            ventana = self.abrir()
        self.assertEqual(
            [p.name for p in ventana.paginas],
            ['001.png', '002.png', 'tomo 2.PDF#001', 'tomo 2.PDF#002'],
        )
        self.assertIn('roto.pdf', aviso.call_args.args[1])
        with mock.patch.object(marcar.simpledialog, 'askinteger', return_value=3):
            ventana.ir_a_pagina()
        self.assertEqual(ventana.imagen.shape, V.bgr_from_file(str(DEMO)).shape)
        self.assertTrue(ventana.cajas, 'el detector propone también en un PDF')
        ventana.pagina_lista()
        self.assertEqual(ventana.indice, 3, 'pasa a la siguiente página del PDF')
        with mock.patch.object(D, 'PROPIOS', Path(self.tmp.name) / 'propios.csv'):
            filas = D.medir_propias(avance=lambda _: None)
        self.assertEqual({f['pagina'] for f in filas}, {'tomo 2.PDF#001'})

    @unittest.skipUnless(pypdfium2, 'pypdfium2 no está instalado')
    def test_un_pdf_suelto_es_una_serie(self):
        # En una carpeta con otras cosas: solo cuentan las páginas de ese PDF.
        tomo = self.crear_pdf(self.carpeta / 'Otra serie.pdf', 3)
        ventana = marcar.Marcador(self.root, tomo, 'Prueba')
        self.assertEqual((ventana.grupo, len(ventana.paginas)), ('Otra serie', 3))
        ventana.pagina_lista()
        self.assertEqual(set(K.leer()['Otra serie']['paginas']), {'Otra serie.pdf#001'})
        self.assertNotIn(str(self.carpeta), K.CAJAS.read_text(encoding='utf8'))
        propios = Path(self.tmp.name) / 'propios.csv'
        with mock.patch.object(D, 'PROPIOS', propios):
            filas = D.medir_propias(avance=lambda _: None)
            self.assertEqual({f['pagina'] for f in filas}, {'Otra serie.pdf#001'})
            tomo.unlink()  # sin el PDF se usan las medidas guardadas
            self.assertEqual(D.medir_propias(avance=lambda _: None), filas)

    def test_dos_carpetas_con_el_mismo_nombre_son_dos_series(self):
        otra = Path(self.tmp.name) / 'otra' / 'mi_manga'
        otra.mkdir(parents=True)
        self.assertEqual(K.registrar_carpeta(self.carpeta), 'mi_manga')
        self.assertEqual(K.registrar_carpeta(otra), 'mi_manga (2)')
        self.assertEqual(K.registrar_carpeta(self.carpeta), 'mi_manga')


if __name__ == '__main__':
    unittest.main()
