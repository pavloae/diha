import numpy as np


class Material:

    def get_stress(self, strain):
        """
            Dada una deformación específica se devuelve la tensión correspondiente según la relación constitutiva
            del material.

        :param strain: Tensión del material, en MPa.
        """
        raise NotImplementedError


class SteelMaterial(Material):

    def __init__(self, fy=420, E=200000):
        """
            Clase para definir la relación constitutiva tensión-deformación en el acero.

        :param fy: Tensión de fluencia del acero, en MPa
        :param E: Módulo de elasticidad del acero, en MPa
        """
        super().__init__()
        self.fy = fy
        self.E = 200000 if E is None else E
        self.limit_strain = 0.005

    def get_stress(self, strain):
        """
            Para una deformación específica devuelve la tensión correspondiente del acero.
        @param strain: Una deformación específica.
        @return: El valor de la tensión correspondiente del acero, en MPa.
        """
        if abs(strain * self.E) < self.fy:
            return strain * self.E
        else:
            return self.fy * np.sign(strain)


class ConcreteMaterial(Material):

    def __init__(self, fpc=20):
        """
            Clase para definir la relación constitutiva tensión-deformación en el hormigón.

        :param fpc: Resistencia característica a compresión del hormigón, en MPa
        """
        super().__init__()
        self.fpc = fpc

        self.epsilon_lim = -0.003
        self._beta1 = None
        self._min_stress = None
        self._epsilon_t = None

        self.limit_strain = -0.003

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
        """
            Deformación del hormigón a partir de la cual se consideran que colabora a compresión.
        @return: El valor de la deformación especifica.
        """
        if not self._epsilon_t:
            self._epsilon_t = (1 - self.beta1) * self.epsilon_lim
        return self._epsilon_t

    @property
    def min_stress(self):
        """
            Tensión máxima de compresión (negativa) que se considera para el bloque de tensiones del hormigón.
        @return: El valor de la tensión de compresión, en MPa.
        """
        if not self._min_stress:
            self._min_stress = -.85 * self.fpc
        return self._min_stress

    def get_stress(self, strain):
        """
            La tensión en el hormigón se adopta igual a 0,85 f’c, y se supone
            uniformemente distribuida en una zona de compresión equivalente, limitada por los
            extremos de la sección transversal, y por una línea recta paralela al eje neutro, a una
            distancia a = β1 · c, a partir de la fibra comprimida con deformación máxima.

        :param strain: Deformación especifica de la fibra de hormigón.
        :return: Tensión de la fibra de hormigón, en MPa.
        """

        if self.epsilon_lim <= strain <= self.epsilon_t:
            return self.min_stress

        return 0
