from unittest import TestCase

import numpy as np

from diha.utils import angle, calc_angle_yz


class TestAngle(TestCase):

    def test_angle_3d(self):
        a = np.array([2, 1, 1])
        b = np.array([5, 0, 1])
        c = np.array([-2, -1, 0])
        d = np.array([-5, 0, -1])

        self.assertAlmostEqual(1.75 * np.pi, calc_angle_yz(b, a))
        self.assertAlmostEqual(np.pi / 4, calc_angle_yz(a, b))

        # El ángulo está dado entre 0 y pi
        self.assertAlmostEqual(3 / 4 * np.pi, calc_angle_yz(a, c))
        self.assertAlmostEqual(np.pi, calc_angle_yz(a, -a))
        self.assertAlmostEqual(1.25 * np.pi, calc_angle_yz(a, d))


class Test(TestCase):

    def test_calc_angle_yz(self):
        self.assertAlmostEqual(0.00 * np.pi, calc_angle_yz(np.array([0, 0, 1]), np.array([0, 0, 1])))
        self.assertAlmostEqual(0.25 * np.pi, calc_angle_yz(np.array([0, 0, 1]), np.array([0, -1, 1])))
        self.assertAlmostEqual(0.50 * np.pi, calc_angle_yz(np.array([0, 0, 1]), np.array([0, -1, 0])))
        self.assertAlmostEqual(0.75 * np.pi, calc_angle_yz(np.array([0, 0, 1]), np.array([0, -1, -1])))
        self.assertAlmostEqual(1.00 * np.pi, calc_angle_yz(np.array([0, 0, 1]), np.array([0, 0, -1])))
        self.assertAlmostEqual(1.25 * np.pi, calc_angle_yz(np.array([0, 0, 1]), np.array([0, 1, -1])))
        self.assertAlmostEqual(1.50 * np.pi, calc_angle_yz(np.array([0, 0, 1]), np.array([0, 1, 0])))
        self.assertAlmostEqual(1.75 * np.pi, calc_angle_yz(np.array([0, 0, 1]), np.array([0, 1, 1])))
