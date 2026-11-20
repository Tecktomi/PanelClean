"""Pruebas de la evaluación: métricas (evaluacion/metricas.py) y pares sintéticos.

Ejecutar desde la raíz del repositorio:
    python -m unittest discover tests
"""

import math
import unittest
from unittest import mock

import numpy as np

from evaluacion import calibrar_selector as C
from evaluacion import evaluar as E
from evaluacion import metricas as M
from evaluacion import muestreo
from evaluacion import sinteticos as S
from panelclean import vision as V


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

    def test_regla_del_selector(self):
        # (entropía en bits, tono oscuro): papel, trama gris y dibujo con tinta.
        self.assertEqual(V.classify_background(0.3, 250)[0], 'Uniforme')
        self.assertEqual(V.classify_background(2.8, 150)[0], 'Estructurado')
        self.assertEqual(V.classify_background(3.5, 5)[0], 'Complejo / revisar')
        self.assertEqual(V.classify_background(0.3, 250, 0.5)[0], 'Revisar máscara')
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


if __name__ == '__main__':
    unittest.main()
