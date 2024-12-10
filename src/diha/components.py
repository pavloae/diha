import math

import numpy as np


class Force:

    tol = 1e-6  # Tolerancia

    def __init__(self, N=0, My=0, Mz=0):
        self.N = N
        self.My = My
        self.Mz = Mz
        self.M = np.array([0, My, Mz])

    def clean(self):
        self.N = 0
        self.My = 0
        self.Mz = 0

    def magnitude(self):
        return math.sqrt(self.N ** 2 + self.My ** 2 + self.Mz ** 2)

    @property
    def e(self):

        if self.N != 0:
            ey = -self.Mz / self.N
            ez = self.My / self.N

        else:
            if self.Mz == 0:
                ey = 0
            elif self.Mz > 0:
                ey = -np.inf
            else:
                ey = np.inf

            if self.My == 0:
                ez = 0
            elif self.My > 0:
                ez = np.inf
            else:
                ez = -np.inf

        return np.array([0, ey, ez])

    def __lt__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        return self.magnitude() < other.magnitude()

    def __le__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        return self.magnitude() <= other.magnitude()

    def __gt__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        return self.magnitude() > other.magnitude()

    def __ge__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        return self.magnitude() >= other.magnitude()

    def __add__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        return Force(self.N + other.N, self.My + other.My, self.Mz + other.Mz)

    def __sub__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        return Force(self.N - other.N, self.My - other.My, self.Mz - other.Mz)

    def __iadd__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        self.N += other.N
        self.My += other.My
        self.Mz += other.Mz
        return self

    def __isub__(self, other):
        if not isinstance(other, Force):
            return NotImplemented
        self.N -= other.N
        self.My -= other.My
        self.Mz -= other.Mz
        return self

    def __repr__(self):
        return f"Force(N={self.N}, My={self.My}, Mz={self.Mz})"

    def __str__(self):
        return f"N = {self.N * 1e-3:5.0f} kN - My = {self.My * 1e-6:5.0f} kNm - Mz = {self.Mz * 1e-6:5.0f} kNm"

    def __eq__(self, other):
        if not isinstance(other, Force):
            return NotImplemented

        return (
                math.isclose(self.N, other.N, abs_tol=self.tol) and
                math.isclose(self.My, other.My, abs_tol=self.tol) and
                math.isclose(self.Mz, other.Mz, abs_tol=self.tol)
        )

    def __mul__(self, factor):
        if isinstance(factor, (int, float)):
            return Force(self.N * factor, self.My * factor, self.Mz * factor)
        raise TypeError("El multiplicador debe ser un escalar (int o float)")

    def __rmul__(self, factor):
        return self.__mul__(factor)  # Reutilizamos la lógica de __mul__


class ForceExt(Force):

    def __init__(self, force, strain_steel, strain_concrete, factor):
        super().__init__(force.N, force.My, force.Mz)
        self.strain_steel = strain_steel
        self.strain_concrete = strain_concrete
        self.factor = factor


class StrainPlane:

    def __init__(self, theta=0, kappa=0, xo=0):
        super().__init__()

        # Angulo entre el vector nn (eje neutro) y el eje z positivo
        self._theta = theta

        # Escalar que define la máxima pendiente del plano de deformaciones (curvatura)
        self._kappa = kappa

        # Escalar que define la coordenada del eje x donde se intersecta con el plano
        self._xo = xo

        # Vector unitario normal al plano de deformaciones
        self._n = None

        # Vector unitario que define la dirección del eje neutro
        self._nn = None

        # Vector que define el desplazamiento del eje neutro.
        self._r = None

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._nn = None
        self._n = None
        self._r = None
        self._theta = theta

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, kappa):
        self._n = None
        self._r = None
        self._kappa = kappa

    @property
    def xo(self):
        return self._xo

    @xo.setter
    def xo(self, xo):
        self._r = None
        self._xo = xo

    @property
    def n(self):
        if self._n is None:
            alpha = np.arctan(self.kappa)
            nxy = np.sin(alpha)
            self._n = np.array([np.cos(alpha), nxy * np.cos(self.theta), nxy * np.sin(self.theta)])
        return self._n

    @property
    def nn(self):
        if self._nn is None:
            self._nn = np.array([0, -np.sin(self.theta), np.cos(self.theta)])

        return self._nn

    @property
    def r(self):
        if self._r is None:
            p = np.cross(self.n, self.nn)
            if p[0] != 0:
                factor = self.xo / p[0]
                self._r = np.array([self.xo, 0, 0]) - factor * p
            else:
                self._r = np.array([0, 0, 0])
        return self._r

    def get_dist_nn_cg(self, point):
        return np.cross(self.nn, point)[0]

    def get_dist_calc(self, point):
        """
            Calcula la distancia de que hay entre el punto y el eje neutro de la sección, o el eje baricéntrico en caso
            de que el plano sea horizontal. La distancia se toma positiva si la fibra se encuentra del lado del eje
            neutro donde una curvatura positiva genera tracción.

        @param point:
        @return:
        """
        s = point - self.r
        return np.cross(self.nn, s)[0]

    def get_strain(self, point):
        """
            Obtiene la deformación específica de un punto determinada por el plano de deformación.

        @param point: Un vector en tres dimensiones sobre el plano YZ
        @return: La deformación específica. Positiva para estiramiento y negativa para acortamiento.
        """

        return self.xo + self.kappa * self.get_dist_nn_cg(point)

    def __repr__(self):
        return f"StrainPlane(theta={self.theta}, kappa={self.kappa}, xo={self.xo})"

    def __str__(self):
        return f"theta={self.theta:8.2f} - \u03BA={1000*self.kappa:.5f}\u2030 - xo={1000*self.xo:.5f}\u2030"


class Stirrups:

    def __init__(self, type=1, number=None, diam=None, sep=None):
        super().__init__()
        self.type = type
        self.number = number
        self.diam = diam
        self.sep = sep
