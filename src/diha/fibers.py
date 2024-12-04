from typing import List

import numpy as np

from .components import Force
from .materials import Material


class Fiber:

    def __init__(self, material: Material, center, area):
        super().__init__()
        self.material: Material = material
        self.center = center
        self.y = center[0]
        self.z = center[1]
        self.area = area

        self.strain = None
        self.stress = None
        self.force = Force()

    def set_strain(self, strain_plane):
        self.strain = strain_plane.get_strain(self.y, self.z)

    def set_force(self):
        self.stress = self.material.get_stress(self.strain)
        N = self.stress * self.area
        self.force = Force(N, N * self.z, -N * self.y)


class RectFiber(Fiber):

    def __init__(self, material, center, dy, dz):
        super().__init__(material, center, dy * dz)
        self.dy = dy
        self.dz = dz

    def get_top_left(self):
        return self.center[0] + self.dy / 2, self.center[1] - self.dz / 2

    def get_top_right(self):
        return self.center[0] + self.dy / 2, self.center[1] + self.dz / 2

    def get_bottom_left(self):
        return self.center[0] - self.dy / 2, self.center[1] - self.dz / 2

    def get_bottom_right(self):
        return self.center[0] - self.dy / 2, self.center[1] + self.dz / 2


class RoundFiber(Fiber):

    def __init__(self, material, center, diam):
        super().__init__(material, center, np.pi / 4 * diam ** 2)
        self.diam = diam


class GroupFiberStatus:

    def __init__(self):
        super().__init__()
        self.force = Force()
        self.max_strain = None
        self.min_strain = None
        self.max_stress = None
        self.min_stress = None

    def update(self, fibers: List[Fiber]):

        self.force.clean()
        self.max_strain = None
        self.min_strain = None
        self.max_stress = None
        self.min_stress = None

        for fiber in fibers:
            self.force += fiber.force

            if not self.max_strain or fiber.strain > self.max_strain:
                self.max_strain = fiber.strain

            if not self.min_strain or fiber.strain < self.min_strain:
                self.min_strain = fiber.strain

            if not self.max_stress or fiber.stress > self.max_stress:
                self.max_stress = fiber.stress

            if not self.min_stress or fiber.stress < self.min_stress:
                self.min_stress = fiber.stress
