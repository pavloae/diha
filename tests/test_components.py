from unittest import TestCase

import numpy as np

from diha.components import StrainPlane, Force

class TestForce(TestCase):

    def test_str(self):

        self.assertEqual(
            str(Force()),
            "N =      0.0 - My =     0.0 - Mz =     0.0"
        )

        self.assertEqual(
            str(Force(N=-250.955, My=300.455, Mz=400.544)),
            "N =   -251.0 - My =   300.5 - Mz =   400.5"
        )

        self.assertEqual(
            str(Force(N=-12345.955, My=3000.455, Mz=4000.544)),
            "N = -12346.0 - My =  3000.5 - Mz =  4000.5"
        )

    def test_theta_m(self):

        self.assertAlmostEqual(0.0 * np.pi, Force(Mz=200).theta_M)
        self.assertAlmostEqual(0.5 * np.pi, Force(My=-200).theta_M)
        self.assertAlmostEqual(1.0 * np.pi, Force(Mz=-200).theta_M)
        self.assertAlmostEqual(1.5 * np.pi, Force(My=200).theta_M)

        self.assertAlmostEqual(1.75 * np.pi, Force(My=200, Mz=200).theta_M)
        self.assertAlmostEqual(1.75 * np.pi, Force(N=200, My=200, Mz=200).theta_M)

        self.assertIsNone(Force().theta_M)
        self.assertIsNone(Force(N=200).theta_M)
        self.assertIsNone(Force(N=200, My=0.001).theta_M)
        self.assertIsNotNone(Force(N=200, My=0.0011).theta_M)

    def test_e(self):

        self.assertAlmostEqual(0.0, Force(N=200).e)
        self.assertAlmostEqual(0.1, Force(N=200, Mz=20).e)
        self.assertAlmostEqual(5.0, Force(N=100, My=400, Mz=300).e)

        self.assertAlmostEqual(np.inf, Force(N=1e-3, My=400, Mz=300).e)
        self.assertGreaterEqual(5e9, Force(N=1e-2, My=400, Mz=300).e)

        self.assertRaises(ArithmeticError, lambda: Force().e)

    def test_eq(self):
        self.assertTrue(Force() == Force())
        self.assertTrue(Force(N=0.0011) == Force(N=0.0009))
        self.assertTrue(Force(N=100.0011) == Force(N=100.0009))
        self.assertTrue(Force(N=100.0015) == Force(N=100.0005))

        self.assertTrue(Force(N=0.0015) == Force(N=0.0025))
        self.assertTrue(Force(N=0.0015) != Force(N=0.0026))

        self.assertTrue(Force(N=10.0015) == Force(N=10.0025))
        self.assertTrue(Force(N=10.0015) != Force(N=10.0026))

class TestStrainPlane(TestCase):

    @staticmethod
    def get_points():
        return [
            np.array([0, 0, 0]),
            np.array([0, 1, 1]),
            np.array([0, 0, 1]),
            np.array([0, -1, 1]),
            np.array([0, -1, 0]),
            np.array([0, -1, -1]),
            np.array([0, 0, -1]),
            np.array([0, 1, -1]),
            np.array([0, 1, 0]),
        ]

    def test_str(self):

        self.assertEqual(
            str(StrainPlane()),
            "theta = 0.00 rad - κ = 0.000 % - xo = 0.000 ‰"
        )

        self.assertEqual(
            str(StrainPlane(theta=np.pi / 2, kappa=0.01, xo=0.005)),
            "theta = 1.57 rad - κ = 1.000 % - xo = 5.000 ‰"
        )

    def test_n(self):

        sp = StrainPlane(theta=0, kappa=0, xo=0)
        self.assertTrue(np.allclose([1, 0, 0], sp.n))

        sp = StrainPlane(theta=0, kappa=1, xo=0)
        self.assertTrue(np.allclose([np.sqrt(2) / 2, np.sqrt(2) / 2, 0], sp.n))

        sp = StrainPlane(theta=0.25 * np.pi, kappa=1, xo=0)
        self.assertTrue(np.allclose([np.sqrt(2) / 2, .5, .5], sp.n))

    def test_nn(self):

        sp = StrainPlane(theta=0, kappa=0, xo=0)
        self.assertTrue(np.allclose([0, 0, 1], sp.nn))

        sp = StrainPlane(theta=0, kappa=1, xo=0)
        self.assertTrue(np.allclose([0, 0, 1], sp.nn))

        sp = StrainPlane(theta=0.25 * np.pi, kappa=1, xo=0)
        self.assertTrue(np.allclose([0, -np.sqrt(2) / 2, np.sqrt(2) / 2], sp.nn))

    def test_r(self):

        sp = StrainPlane(theta=0, kappa=0, xo=0)
        self.assertTrue(np.allclose([0, 0, 0], sp.r))

        sp = StrainPlane(theta=0, kappa=1, xo=0)
        self.assertTrue(np.allclose([0, 0, 0], sp.r))

        sp = StrainPlane(theta=0, kappa=1, xo=1)
        self.assertTrue(np.allclose([0, 1, 0], sp.r))

        sp = StrainPlane(theta=0, kappa=1, xo=-1)
        self.assertTrue(np.allclose([0, -1, 0], sp.r))

        sp = StrainPlane(theta=0.25 * np.pi, kappa=0.2, xo=0.2)
        self.assertTrue(np.allclose([0, np.sqrt(2) / 2, np.sqrt(2) / 2], sp.r))

        sp = StrainPlane(theta=0.5 * np.pi, kappa=0.5, xo=4)
        self.assertTrue(np.allclose([0, 0, 8], sp.r))

        sp = StrainPlane(theta=0.5 * np.pi, kappa=0.5, xo=-4)
        self.assertTrue(np.allclose([0, 0, -8], sp.r))

    def test_strain_plane_n(self):

        # El vector normal al plano por defecto está sobre el eje x
        sp = StrainPlane()
        self.assertTrue(np.array_equal(sp.n, [1, 0, 0]))

        # Si se define un giro y un desplazamiento el vector normal sigue estando en el eje x
        sp = StrainPlane(theta=np.pi / 4, xo=0.01)
        self.assertTrue(np.array_equal(sp.n, [1, 0, 0]))

        # Si se define una inclinación el vector normal deja de estar en la dirección del eje x

        # Curvatura positiva de 30º
        alpha = np.pi / 6
        kappa = np.tan(alpha)
        self.assertAlmostEqual(kappa, .577350269)

        sp = StrainPlane(kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, 0.5, 0.0]))

        sp = StrainPlane(theta=.25 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, 0.35355, 0.35355]))

        sp = StrainPlane(theta=.50 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .0, .5]))

        sp = StrainPlane(theta=.75 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -0.35355, 0.35355]))

        sp = StrainPlane(theta=1.00 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -.5, .0]))

        sp = StrainPlane(theta=1.25 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -.35355, -.35355]))

        sp = StrainPlane(theta=1.50 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .0, -.5]))

        sp = StrainPlane(theta=1.75 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .35355, -.35355]))

        sp = StrainPlane(theta=2.00 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .5, .0]))

        # Se gira el ángulo 180 para simular Curvatura negativa de 30º

        sp = StrainPlane(theta=1.00 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -0.5, 0.0]))

        sp = StrainPlane(theta=1.25 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -0.35355, -0.35355]))

        sp = StrainPlane(theta=1.50 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .0, -.5]))

        sp = StrainPlane(theta=1.75 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, 0.35355, -0.35355]))

        sp = StrainPlane(theta=0.00 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .5, .0]))

        sp = StrainPlane(theta=0.25 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .35355, .35355]))

        sp = StrainPlane(theta=0.50 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .0, .5]))

        sp = StrainPlane(theta=0.75 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -.35355, .35355]))

        sp = StrainPlane(theta=1.00 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -.5, .0]))

    def test_strain_plane_nn(self):
        sp = StrainPlane(theta=.25 * np.pi, kappa=1)
        nn = np.array([0, -1, 1])
        nn = nn / np.linalg.norm(nn)
        self.assertTrue(np.allclose(sp.nn, nn))

        sp = StrainPlane(theta=.75 * np.pi, kappa=1)
        nn = np.array([0, -1, -1])
        nn = nn / np.linalg.norm(nn)
        self.assertTrue(np.allclose(sp.nn, nn))

        sp = StrainPlane(theta=1.25 * np.pi, kappa=1)
        nn = np.array([0, 1, -1])
        nn = nn / np.linalg.norm(nn)
        self.assertTrue(np.allclose(sp.nn, nn))

        sp = StrainPlane(theta=1.75 * np.pi, kappa=1)
        nn = np.array([0, 1, 1])
        nn = nn / np.linalg.norm(nn)
        self.assertTrue(np.allclose(sp.nn, nn))

    def test_strain_plane_get_strain(self):

        for xo in [-1, 0, 1]:
            sp = StrainPlane(xo=xo)
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, 0])), xo))
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, -1, 1])), xo))
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, -1, -1])), xo))
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, 1, -1])), xo))
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, 1, 1])), xo))

        # Curvatura alrededor del eje z
        k = 0.1
        sp = StrainPlane(theta=0, kappa=k, xo=0)
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, 0])), 0))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, 1])), 0))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, -1, 0])), k))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, -1])), 0))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, 1, 0])), -k))

        # Curvatura alrededor del eje y
        sp = StrainPlane(theta=1.5 * np.pi, kappa=k, xo=0)
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, 0])), 0))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, 1])), k))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, -1, 0])), 0))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, -1])), -k))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, 1, 0])), 0))

        # Curvatura alrededor de un eje a 45º
        sp = StrainPlane(theta=np.pi / 4, kappa=0.1, xo=0)
        e = np.sqrt(2) / 2
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, 0])), 0))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, -e, e])), 0))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, -e, -e])), 0.1))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, e, -e])), 0))
        self.assertTrue(np.allclose(sp.get_strain(np.array([0, e, e])), -0.1))

    def test_get_dist_nn(self):

        point_0 = np.array([0, 0, 0])
        point_1 = np.array([0, 1, 1])
        point_2 = np.array([0, 0, 1])
        point_3 = np.array([0, -1, 1])
        point_4 = np.array([0, -1, 0])
        point_5 = np.array([0, -1, -1])
        point_6 = np.array([0, 0, -1])
        point_7 = np.array([0, 1, -1])
        point_8 = np.array([0, 1, 0])

        kappa, xo = 0, 0

        sp = StrainPlane(theta=0, kappa=kappa, xo=xo)
        for point in [point_1, point_7, point_8]:
            self.assertAlmostEqual(1, sp.get_dist_nn(point))
        for point in [point_0, point_2, point_6]:
            self.assertAlmostEqual(0, sp.get_dist_nn(point))
        for point in [point_3, point_4, point_5]:
            self.assertAlmostEqual(-1, sp.get_dist_nn(point))

        sp = StrainPlane(theta=0.25 * np.pi, kappa=kappa, xo=xo)
        for point in [point_1]:
            self.assertAlmostEqual(np.sqrt(2), sp.get_dist_nn(point))
        for point in [point_2, point_8]:
            self.assertAlmostEqual(np.sqrt(2) / 2, sp.get_dist_nn(point))
        for point in [point_0, point_3, point_7]:
            self.assertAlmostEqual(0, sp.get_dist_nn(point))
        for point in [point_4, point_6]:
            self.assertAlmostEqual(-np.sqrt(2) / 2, sp.get_dist_nn(point))
        for point in [point_5]:
            self.assertAlmostEqual(-np.sqrt(2), sp.get_dist_nn(point))

        sp = StrainPlane(theta=0.5 * np.pi, kappa=kappa, xo=xo)
        for point in [point_1, point_2, point_3]:
            self.assertAlmostEqual(1, sp.get_dist_nn(point))
        for point in [point_0, point_4, point_8]:
            self.assertAlmostEqual(0, sp.get_dist_nn(point))
        for point in [point_5, point_6, point_7]:
            self.assertAlmostEqual(-1, sp.get_dist_nn(point))

        sp = StrainPlane(theta=np.pi, kappa=kappa, xo=xo)
        for point in [point_3, point_4, point_5]:
            self.assertAlmostEqual(1, sp.get_dist_nn(point))
        for point in [point_0, point_2, point_6]:
            self.assertAlmostEqual(0, sp.get_dist_nn(point))
        for point in [point_1, point_7, point_8]:
            self.assertAlmostEqual(-1, sp.get_dist_nn(point))

        sp = StrainPlane(theta=1.5 * np.pi, kappa=kappa, xo=xo)
        for point in [point_5, point_6, point_7]:
            self.assertAlmostEqual(1, sp.get_dist_nn(point))
        for point in [point_0, point_4, point_8]:
            self.assertAlmostEqual(0, sp.get_dist_nn(point))
        for point in [point_1, point_2, point_3]:
            self.assertAlmostEqual(-1, sp.get_dist_nn(point))

        h = 2
        epsilon = -0.001
        xo = -epsilon
        kappa = -epsilon / (h / 4)
        sp = StrainPlane(theta=0, kappa=kappa, xo=xo)
        for point in [point_1, point_7, point_8]:
            self.assertAlmostEqual(.25 * h, sp.get_dist_nn(point))
        for point in [point_0, point_2, point_6]:
            self.assertAlmostEqual(-.25 * h, sp.get_dist_nn(point))
        for point in [point_3, point_4, point_5]:
            self.assertAlmostEqual(-.75 * h, sp.get_dist_nn(point))

        h = 2 * np.sqrt(2)  # 1.4142135623730951
        epsilon = -0.001
        xo = -epsilon
        kappa = -epsilon / (h / 4)  # 0.00282842712474619
        sp = StrainPlane(theta=.25 * np.pi, kappa=kappa, xo=xo)

        for point in [point_1]:
            self.assertAlmostEqual(.25 * h, sp.get_dist_nn(point))
        for point in [point_2, point_8]:
            self.assertAlmostEqual(0, sp.get_dist_nn(point))
        for point in [point_0, point_3, point_7]:
            self.assertAlmostEqual(-.25 * h, sp.get_dist_nn(point))
        for point in [point_4, point_6]:
            self.assertAlmostEqual(-.5 * h, sp.get_dist_nn(point))
        for point in [point_5]:
            self.assertAlmostEqual(-.75 * h, sp.get_dist_nn(point))

    def test_get_dist_nn_cg(self):

        point_0 = np.array([0, 0, 0])
        point_1 = np.array([0, 1, 1])
        point_2 = np.array([0, 0, 1])
        point_3 = np.array([0, -1, 1])
        point_4 = np.array([0, -1, 0])
        point_5 = np.array([0, -1, -1])
        point_6 = np.array([0, 0, -1])
        point_7 = np.array([0, 1, -1])
        point_8 = np.array([0, 1, 0])

        for kappa in [-0.1, 0.0, 0.1]:
            for xo in [-1, 0, 1]:

                sp = StrainPlane(theta=0, kappa=kappa, xo=xo)
                for point in [point_1, point_7, point_8]:
                    self.assertAlmostEqual(1, sp.get_dist_nn_cg(point))
                for point in [point_0, point_2, point_6]:
                    self.assertAlmostEqual(0, sp.get_dist_nn_cg(point))
                for point in [point_3, point_4, point_5]:
                    self.assertAlmostEqual(-1, sp.get_dist_nn_cg(point))

                sp = StrainPlane(theta=0.25 * np.pi, kappa=kappa, xo=xo)
                for point in [point_1]:
                    self.assertAlmostEqual(np.sqrt(2), sp.get_dist_nn_cg(point))
                for point in [point_2, point_8]:
                    self.assertAlmostEqual(np.sqrt(2) / 2, sp.get_dist_nn_cg(point))
                for point in [point_0, point_3, point_7]:
                    self.assertAlmostEqual(0, sp.get_dist_nn_cg(point))
                for point in [point_4, point_6]:
                    self.assertAlmostEqual(-np.sqrt(2) / 2, sp.get_dist_nn_cg(point))
                for point in [point_5]:
                    self.assertAlmostEqual(-np.sqrt(2), sp.get_dist_nn_cg(point))

                sp = StrainPlane(theta=0.5 * np.pi, kappa=kappa, xo=xo)
                for point in [point_1, point_2, point_3]:
                    self.assertAlmostEqual(1, sp.get_dist_nn_cg(point))
                for point in [point_0, point_4, point_8]:
                    self.assertAlmostEqual(0, sp.get_dist_nn_cg(point))
                for point in [point_5, point_6, point_7]:
                    self.assertAlmostEqual(-1, sp.get_dist_nn_cg(point))

                sp = StrainPlane(theta=np.pi, kappa=kappa, xo=xo)
                for point in [point_3, point_4, point_5]:
                    self.assertAlmostEqual(1, sp.get_dist_nn_cg(point))
                for point in [point_0, point_2, point_6]:
                    self.assertAlmostEqual(0, sp.get_dist_nn_cg(point))
                for point in [point_1, point_7, point_8]:
                    self.assertAlmostEqual(-1, sp.get_dist_nn_cg(point))

                sp = StrainPlane(theta=1.5 * np.pi, kappa=kappa, xo=xo)
                for point in [point_5, point_6, point_7]:
                    self.assertAlmostEqual(1, sp.get_dist_nn_cg(point))
                for point in [point_0, point_4, point_8]:
                    self.assertAlmostEqual(0, sp.get_dist_nn_cg(point))
                for point in [point_1, point_2, point_3]:
                    self.assertAlmostEqual(-1, sp.get_dist_nn_cg(point))

    def test_get_strain(self):
        points = self.get_points()
        sp = StrainPlane(theta=0, kappa=0, xo=0)
        for point in points:
            self.assertAlmostEqual(0, sp.get_strain(point))

        sp = StrainPlane(theta=0, kappa=0, xo=0.001)
        for point in points:
            self.assertAlmostEqual(0.001, sp.get_strain(point))

        sp = StrainPlane(theta=0, kappa=0, xo=-0.001)
        for point in points:
            self.assertAlmostEqual(-0.001, sp.get_strain(point))

        sp = StrainPlane(theta=0, kappa=0.001, xo=0)
        for point in [points[1], points[7], points[8]]:
            self.assertAlmostEqual(-0.001, sp.get_strain(point))
        for point in [points[0], points[2], points[6]]:
            self.assertAlmostEqual(0, sp.get_strain(point))
        for point in [points[3], points[4], points[5]]:
            self.assertAlmostEqual(0.001, sp.get_strain(point))

        h = 2 * np.sqrt(2)
        kappa = 0.001
        sp = StrainPlane(theta=0.25 * np.pi, kappa=kappa, xo=0)
        for point in [points[1]]:
            self.assertAlmostEqual(-kappa * h / 2, sp.get_strain(point))
        for point in [points[2], points[8]]:
            self.assertAlmostEqual(-kappa * h / 4, sp.get_strain(point))
        for point in [points[0], points[3], points[7]]:
            self.assertAlmostEqual(0, sp.get_strain(point))
        for point in [points[4], points[6]]:
            self.assertAlmostEqual(kappa * h / 4, sp.get_strain(point))
        for point in [points[5]]:
            self.assertAlmostEqual(kappa * h / 2, sp.get_strain(point))

        kappa = 0.001 / (h / 4)
        sp = StrainPlane(theta=0.25 * np.pi, kappa=kappa, xo=0.001)
        for point in [points[1]]:
            self.assertAlmostEqual(-0.001, sp.get_strain(point))
        for point in [points[2], points[8]]:
            self.assertAlmostEqual(0, sp.get_strain(point))
        for point in [points[0], points[3], points[7]]:
            self.assertAlmostEqual(0.001, sp.get_strain(point))
        for point in [points[4], points[6]]:
            self.assertAlmostEqual(0.002, sp.get_strain(point))
        for point in [points[5]]:
            self.assertAlmostEqual(0.003, sp.get_strain(point))

