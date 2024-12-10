import numpy as np


class Material:

    def get_stress(self, strain):
        """
            Dada una deformación específica se devuelve la tensión correspondiente según la relación constitutiva
            del material.

        :param strain:
        """
        raise NotImplementedError


class SteelMaterial(Material):

    def __init__(self, fy=420, E=200000):
        super().__init__()
        self.fy = fy
        self.E = E
        self.limit_strain = 0.005

    def get_stress(self, strain):
        if abs(strain * self.E) < self.fy:
            return strain * self.E
        else:
            return self.fy * np.sign(strain)


class ConcreteMaterial(Material):

    def __init__(self, fpc=20):
        """
            Clase para definir la relación constitutiva tensión-deformación en el hormigón.

        :param fpc: Resistencia característica a compresión del hormigón.
        """
        super().__init__()
        self.fpc = fpc

        self.epsilon_lim = -0.003
        self._beta1 = None
        self._min_stress = None
        self._epsilon_t = None

        self.limit_strain = -0.003

        self.factor = 1

    @property
    def beta1(self):
        """
            Calcula el factor que relaciona la altura del bloque de tensiones de compresión rectangular
            equivalente con la profundidad del eje neutro. Ver el artículo 10.2.7.3

        :return: Un escalar adimensional
        """
        if not self._beta1:
            if self.fpc <= 30:
                self._beta1 = 0.85
            else:
                self._beta1 = max(0.65, 0.85 - 0.05 * (self.fpc - 30) / 7)
        return self._beta1

    @property
    def epsilon_t(self):
        if not self._epsilon_t:
            self._epsilon_t = (1 - self.beta1) * self.epsilon_lim
        return self._epsilon_t

    @property
    def min_stress(self):
        if not self._min_stress:
            self._min_stress = -.85 * self.fpc
        return self._min_stress

    def get_stress(self, strain):
        """
            La tensión en el hormigón se adopta igual a 0,85 f’c , y se supone
            uniformemente distribuida en una zona de compresión equivalente, limitada por los
            extremos de la sección transversal, y por una línea recta paralela al eje neutro, a una
            distancia a = β1 · c, a partir de la fibra comprimida con deformación máxima.

        :param strain: Deformación especifica de la fibra de hormigón.
        :return: Tensión de la fibra de hormigón, en MPa.
        """

        if strain <= self.epsilon_t:
            return self.factor * self.min_stress

        return 0
