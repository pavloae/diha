from unittest import TestCase

import numpy as np

from diha.components import StrainPlane, Force


class TestStrainPlane(TestCase):

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

        # Curvatura negativa de 30º
        alpha = -np.pi / 6
        kappa = np.tan(alpha)
        self.assertAlmostEqual(kappa, -.577350269)

        sp = StrainPlane(kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -0.5, 0.0]))

        sp = StrainPlane(theta=.25 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -0.35355, -0.35355]))

        sp = StrainPlane(theta=.50 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .0, -.5]))

        sp = StrainPlane(theta=.75 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, 0.35355, -0.35355]))

        sp = StrainPlane(theta=1.00 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .5, .0]))

        sp = StrainPlane(theta=1.25 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .35355, .35355]))

        sp = StrainPlane(theta=1.50 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, .0, .5]))

        sp = StrainPlane(theta=1.75 * np.pi, kappa=kappa)
        self.assertTrue(np.allclose(sp.n, [0.86603, -.35355, .35355]))

        sp = StrainPlane(theta=2.00 * np.pi, kappa=kappa)
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
        for k in [-0.1, 0.1]:
            sp = StrainPlane(theta=0, kappa=k, xo=0)
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, 0])), 0))
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, 1])), 0))
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, -1, 0])), k))
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, 0, -1])), 0))
            self.assertTrue(np.allclose(sp.get_strain(np.array([0, 1, 0])), -k))

        # Curvatura alrededor del eje y
        for k in [-0.1, 0.1]:
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


class TestForce(TestCase):

    def test_theta_m(self):
        force = Force(N=-7501670, My=0, Mz=0)
        print(force.theta_M)
