from matplotlib import pyplot as plt

from .fibers import RectFiber, RoundFiber
from .interaction_diagram import ReinforcementConcreteSection
from .materials import ConcreteMaterial


class RectangularRCSection(ReinforcementConcreteSection):

    def __init__(self, concrete, steel, b, h, bars, div_y=None, div_z=None, N=0, My=0, Mz=0, max_initial_strain=0.001):
        super().__init__(concrete, steel, bars, N, My, Mz, max_initial_strain)
        self.b = b
        self.h = h

        if not div_z and not div_y:
            if h > b:
                div_z = 20
            else:
                div_y = 20

        if div_z and not div_y:
            div_y = int(h / b * div_z) & -2
        elif div_y and not div_z:
            div_y = int(h / b * div_z) & -2

        self.div_y = div_y
        self.div_z = div_z

    def _build_concrete_fibers(self):

        delta_y = self.h / self.div_y
        delta_z = self.b / self.div_z

        self.concrete_fibers = []
        for i in range(self.div_y):
            for j in range(self.div_z):

                y_inicial = -self.h / 2 + i * delta_y
                z_inicial = -self.b / 2 + j * delta_z

                self.concrete_fibers.append(
                    RectFiber(
                        self.concrete, (y_inicial + delta_y / 2, z_inicial + delta_z / 2), delta_y, delta_z
                    )
                )

        concrete_negative = ConcreteMaterial()
        concrete_negative.factor = -1

        for fiber in self.steel_fibers:
            self.concrete_fibers.append(
                RoundFiber(concrete_negative, fiber.center, fiber.diam)
            )

    # PlotterMixin

    def _plot_contour(self, ax):
        ax.add_patch(plt.Rectangle((-self.b / 2, -self.h / 2), self.b, self.h, fill=False, edgecolor='black', lw=2))

    def _set_limits(self, ax):
        ax.set_xlim(self.b / 2 + 50, -self.b / 2 - 50)
        ax.set_ylim(-self.h / 2 - 50, self.h / 2 + 50)
