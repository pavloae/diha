from .fibers import RectFiber, RoundFiber
from .interaction_diagram import ReinforcementConcreteSection
from .materials import ConcreteMaterial


class RectangularRCSection(ReinforcementConcreteSection):

    def __init__(self, concrete, steel, b, h, bars, stirrups=None, div_y=None, div_z=None):
        super().__init__(concrete, steel, bars, stirrups)
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

        self._concrete_fibers.clear()
        for i in range(self.div_y):
            for j in range(self.div_z):

                y_inicial = -self.h / 2 + i * delta_y
                z_inicial = -self.b / 2 + j * delta_z

                self._concrete_fibers.append(
                    RectFiber(
                        self.concrete, (y_inicial + delta_y / 2, z_inicial + delta_z / 2), delta_y, delta_z
                    )
                )

        # Se descuentan las armaduras para el cálculo de las fuerzas generadas por el hormigón a compresión
        for fiber in self.steel_fibers:
            self.concrete_fibers.append(
                RoundFiber(ConcreteMaterial(), fiber.center, fiber.diam).set_negative()
            )
