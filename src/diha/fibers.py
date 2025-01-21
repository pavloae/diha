from typing import List

import numpy as np
from matplotlib import pyplot as plt, patches

from .components import Force
from .geometry import Point2D
from .materials import Material


class Fiber:

    def __init__(self, material: Material, center, area):
        """
            Representación de una fibra mediante las propiedades de un material, un área y las coordenadas de su baricentro.

        @param material: Objeto con las propiedades constitutivas del material.
        @param center: Un array de coordenadas representando el baricentro de la fibra [yg, zg], en mm.
        @param area: El área de la fibra, en mm²
        """

        super().__init__()
        self.material: Material = material
        self.center = Point2D(*center)

        self.y = center[0]
        self.z = center[1]
        # self.point = np.array([0, self.y, self.z])
        self._area = area

        self.distance_nn = None
        self.distance_nn_cg = None
        self._strain = None
        self._stress = None
        self._force = None

    @property
    def strain(self):
        return self._strain

    @strain.setter
    def strain(self, strain):
        self._stress = None
        self._force = None
        self._strain = strain

    @property
    def area(self):
        return self._area

    @property
    def stress(self):
        if not self._stress:
            self._stress = self.material.get_stress(self.strain)
        return self._stress

    @property
    def force(self):
        if not self._force:
            N = self.stress * self.area
            self._force = Force(N, N * self.z, -N * self.y)

        return self._force

    def plot(self, ax, color=None):
        pass

    def set_negative(self):
        if self._area > 0:
            self._area *= -1
        return self


class RectFiber(Fiber):

    def __init__(self, material, center, dy, dz):
        super().__init__(material, center, dy * dz)
        self.dy = dy
        self.dz = dz

    def plot(self, ax, color=None):
        y0, z0 = self.center[0] - self.dy / 2, self.center[1] - self.dz / 2
        rect = patches.Rectangle(
            (z0, y0), self.dz, self.dy, edgecolor="gray", facecolor="lightblue", alpha=0.5
        )
        rect.set_facecolor(color)
        ax.add_patch(rect)


class RoundFiber(Fiber):

    def __init__(self, material, center, diam):
        super().__init__(material, center, np.pi / 4 * diam ** 2)
        self.diam = diam

    def plot(self, ax, color=None):
        y, z = self.center
        circle = plt.Circle((z, y), self.diam / 2, color='red', alpha=0.7)
        circle.set_facecolor(color)
        ax.add_patch(circle)


class GroupFiberStatus:

    def __init__(self):
        super().__init__()
        self.force = Force()

    def update(self, fibers: List[Fiber]):

        self.force = Force()

        for fiber in fibers:
            self.force += fiber.force
