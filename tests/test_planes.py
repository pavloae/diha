import logging
from unittest import TestCase

import numpy as np

from src.diha.fibers import RoundFiber
from diha.components import StrainPlane
from src.diha.materials import ConcreteMaterial, SteelMaterial
from src.diha.sections import RectangularRCSection

logging.basicConfig(level=logging.DEBUG)

class TestInteractionDiagram(TestCase):

    def get_section(self):
        concrete = ConcreteMaterial()
        steel = SteelMaterial()

        b = 500
        h = 750
        rec = 50

        diam = 16

        bars = [
            RoundFiber(steel, (h / 2 - rec, b / 2 - rec), diam), RoundFiber(steel, (h / 2 - rec, -b / 2 + rec), diam),
            RoundFiber(steel, (0, b / 2 - rec), diam), RoundFiber(steel, (0, -b / 2 + rec), diam),
            RoundFiber(steel, (-h / 2 + rec, b / 2 - rec), diam), RoundFiber(steel, (-h / 2 + rec, -b / 2 + rec), diam),
        ]

        section = RectangularRCSection(concrete, steel, b, h, bars)

        return section

    def test_strain_plane_nn(self):


        x = 1

        sp = StrainPlane([x, 1, 1])
        nn = np.array([0, -1, 1])
        nn = nn / np.linalg.norm(nn)
        self.assertTrue(np.allclose(sp._nn, nn))

        sp = StrainPlane([x, -1, 1])
        nn = np.array([0, -1, -1])
        nn = nn / np.linalg.norm(nn)
        self.assertTrue(np.allclose(sp._nn, nn))

        sp = StrainPlane([x, -1, -1])
        nn = np.array([0, 1, -1])
        nn = nn / np.linalg.norm(nn)
        self.assertTrue(np.allclose(sp._nn, nn))

        sp = StrainPlane([x, 1, -1])
        nn = np.array([0, 1, 1])
        nn = nn / np.linalg.norm(nn)
        self.assertTrue(np.allclose(sp._nn, nn))

    def test_strain_plane_rotate(self):
        sp = StrainPlane()

        x = 1

        sp.n = [x, 1, 1]

        sp.rotate(np.pi / 2)
        nr = np.array([x, -1, 1])
        nr = nr / np.linalg.norm(nr)
        self.assertTrue(np.allclose(sp.n, nr))

        sp.rotate(np.pi / 2)
        nr = np.array([x, -1, -1])
        nr = nr / np.linalg.norm(nr)
        self.assertTrue(np.allclose(sp.n, nr))

        sp.rotate(np.pi / 2)
        nr = np.array([x, 1, -1])
        nr = nr / np.linalg.norm(nr)
        self.assertTrue(np.allclose(sp.n, nr))

        sp.rotate(np.pi / 2)
        nr = np.array([x, 1, 1])
        nr = nr / np.linalg.norm(nr)
        self.assertTrue(np.allclose(sp.n, nr))

    def test_strain_plane_inclinate(self):

        n_list = []
        n_rotated_45 = []
        for octant in range(1, 5):
            nx = 2 ** .5 if octant <= 4 else -2 ** .5
            ny = 1 if octant in [1, 4, 5, 8] else -1
            nz = 1 if octant in [1, 2, 5, 6] else -1
            n = [nx, ny, nz]
            n_list.append([nx, ny, nz])

            nrx = 0
            nry = 1 if octant in [1, 4, 5, 8] else -1
            nrz = 1 if octant in [1, 2, 5, 6] else -1

            if octant >= 5:
                nrx = -1
                nry = 0
                nrz = 0

            nr = np.array([nrx, nry, nrz])
            nr = nr / np.linalg.norm(nr)
            n_rotated_45.append(nr)


        for n, nr in zip(n_list, n_rotated_45):
            sp = StrainPlane(n)
            sp.inclinate(np.pi / 4)
            self.assertTrue(np.allclose(sp.n, nr))

    def test_strain_plane_move(self):
        sp = StrainPlane()

        test_points = [
            [1, -1], [1, 0], [1, 1],
            [0, -1], [0, 0], [0, 1],
            [-1, -1], [-1, 0], [-1, 1]
        ]

        for point in test_points:
            strain = sp.get_strain(*point)
            self.assertAlmostEqual(0.0, strain)

        sp.move(0.001)
        for point in test_points:
            strain = sp.get_strain(*point)
            self.assertAlmostEqual(0.001, strain)

        sp = StrainPlane()
        k = 0.001
        ang = np.arctan(k)
        test_strain = [
            2 * k, k, 0,
            k, 0, -k,
            0, -k, -2 * k
        ]

        sp.inclinate(ang, axis='y')
        sp.inclinate(-ang, axis='z')
        for point, strain_test in zip(test_points, test_strain):
            strain = sp.get_strain(point[1], point[0])
            self.assertAlmostEqual(strain_test, strain, delta=0.000001)

        epsilon = 0.001
        sp.move(0.001)
        for point, strain_test in zip(test_points, test_strain):
            strain = sp.get_strain(point[1], point[0])
            self.assertAlmostEqual(strain_test + epsilon, strain, delta=0.000001)
