from .fibers import RectFiber, RoundFiber
from .calc import ReinforcementConcreteSectionBase


class RectangularRCSectionBase(ReinforcementConcreteSectionBase):

    def __init__(self, concrete, steel, b, h, bars, stirrups=None, div_y=None, div_z=None):
        """
            Clase apra la construcciónn de una sección de hormigón armado rectangular.
        @param concrete: Material que define las propiedades del hormigón
        @param steel: Material que define las propiedades del acero
        @param b: Ancho de la sección
        @param h: Altura de la sección
        @param bars: Array con las barras de acero
        @param stirrups: Propiedades de los estribos
        @param div_y: Número de divisiones horizontales a utilizar
        @param div_z: Número de divisiones verticales a utilizar
        """
        super().__init__(concrete, steel, bars, stirrups)
        self.b = b
        self.h = h

        if not div_z and not div_y:
            if h > b:
                div_z = 32
            else:
                div_y = 32

        if div_z and not div_y:
            div_y = int(h / b * div_z) & -2
        elif div_y and not div_z:
            div_z = int(b / h * div_y) & -2

        self.div_y = div_y
        self.div_z = div_z

    def increase_resolution(self, factor):
        self.div_y = self.div_y * factor
        self.div_z = self.div_z * factor
        self.build(force=True)

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
                RoundFiber(self.concrete, fiber.center, fiber.diam).set_negative()
            )
