import math

import numpy as np

from diha.utils import calc_angle_yz, norm_ang


class Force:

    tol = 1e-6  # Tolerancia

    def __init__(self, N=0, My=0, Mz=0):
        self.N = N
        self.My = My
        self.Mz = Mz
        self.M = np.array([0, My, Mz])
        self._theta_M = None
        self._e = None

    def magnitude(self):
        return math.sqrt(self.N ** 2 + self.My ** 2 + self.Mz ** 2)

    @property
    def theta_M(self):
        """
            Ángulo que forma el vector de momentos con respecto al eje "z" positivo medido en sentido antihorario.
        @return: Un real entre 0 y 2 pi o None si no existe momento.
        """
        if not self._theta_M:
            if not np.isclose(0, np.linalg.norm(self.M), atol=1e-6):
                self._theta_M = calc_angle_yz(np.array([0, 0, 1]), self.M)
        return self._theta_M

    @property
    def e(self):
        if not self._e:
            self._e = np.linalg.norm(self.M) / self.N if self.N != 0 else np.inf
        return self._e

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

    def __init__(self, force, strain_steel):
        super().__init__(force.N, force.My, force.Mz)
        self.strain_steel = strain_steel


class StrainPlane:

    def __init__(self, theta=0, kappa=0, xo=0):
        """
            Define un plano de deformaciones

        @param theta: Ángulo que forma el eje positivo de giro del plano respecto al eje "z" positivo medido en sentido
        antihorario. Un valor real entre 0 y 2 pi.
        @param kappa: Pendiente del plano de deformaciones. Valor real positivo entre 0 (horizontal) y pi/2 (vertical).
        @param xo: Desplazamiento del plano de deformaciones del centro de coordenadas en sentido vertical.
        """
        super().__init__()

        # Angulo entre el vector nn (eje neutro) y el eje "z" positivo
        self._theta = theta

        # Escalar que define la máxima pendiente del plano de deformaciones (curvatura)
        self._kappa = kappa

        # Escalar que define la coordenada del eje "x" donde se intersecta con el plano
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
        self._theta = norm_ang(theta)

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
            nyz = np.sin(alpha)
            self._n = np.array([np.cos(alpha), nyz * np.cos(self.theta), nyz * np.sin(self.theta)])
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
                pe = factor * p
                self._r = np.array([self.xo, 0, 0]) - pe
            else:
                self._r = np.array([0, 0, 0])
        return self._r

    def get_dist_nn_cg(self, point):
        """
            Calcula la distancia que hay entre el punto y el eje baricéntrico paralelo al eje neutro.
        @param point:
        @return:
        """
        return np.cross(point, self.nn)[0]

    def get_dist_nn(self, point):
        """
            Calcula la distancia de que hay entre el punto y el eje neutro de la sección, o el eje baricéntrico en caso
            de que el plano sea horizontal. La distancia se toma positiva si la fibra se encuentra del lado del eje
            neutro donde una curvatura positiva genera compresión.

        @param point:
        @return:
        """
        s = point - self.r
        return np.cross(s, self.nn)[0]

    def get_strain(self, point):
        """
            Obtiene la deformación específica de un punto determinada por el plano de deformación.

        @param point: Un vector en tres dimensiones sobre el plano YZ
        @return: La deformación específica. Positiva para estiramiento y negativa para acortamiento.
        """

        return self.xo - self.kappa * self.get_dist_nn_cg(point)

    def __repr__(self):
        return f"StrainPlane(theta={self.theta}, kappa={self.kappa}, xo={self.xo})"

    def __str__(self):
        return f"theta={self.theta:8.2f} - \u03BA={1000*self.kappa:.5f}\u2030 - xo={1000*self.xo:.5f}\u2030"


class Stirrups:

    def __init__(self, stirrup_type=1, number=None, diam=None, sep=None):
        super().__init__()
        self.stirrup_type = stirrup_type
        self.number = number
        self.diam = diam
        self.sep = sep
