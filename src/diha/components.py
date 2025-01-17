import math

import numpy as np

from diha.geometry import Point2D
from diha.utils import calc_angle_yz, norm_ang


class Force:

    # Por defecto se mantiene una precisión de 1N sin importar la magnitud de la fuerza
    atol = 1e-3  # Tolerancia absoluta
    rtol = 0.0

    def __init__(self, N=0.0, My=0.0, Mz=0.0):
        """
            Representación de una fuerza para el análisis de secciones de hormigón armado sometidas a esfuerzos normales
            de flexión compuesta siguiendo los lineamientos de CIRSOC 201. El eje "x" está dirigido hacia afuera del
            plano, el eje y hacia arriba y el eje z hacia la izquierda, utilizando la regla de la mano derecha.

        @param N: Fuerza axil de compresión (negativa) o tracción (positiva), en N.
        @param My: Momento flector alrededor del eje "y", en Nmm.
        @param Mz: Momento flector alrededor del eje "z", en Nmm.
        """
        self.N = N
        self.My = My
        self.Mz = Mz

    @property
    def M(self):
        return np.array([0, self.My, self.Mz])

    @property
    def mod_M(self):
        return np.linalg.norm(self.M)

    @property
    def theta_M(self):
        """
            Ángulo que forma el vector de momentos con respecto al eje "z" positivo medido en sentido antihorario.
        @return: Un real entre 0 y 2 pi o None si no existe momento.
        """
        if not math.isclose(0.0, self.mod_M, abs_tol=self.atol, rel_tol=self.rtol):
            return calc_angle_yz(np.array([0, 0, 1]), self.M)
        return None

    @property
    def e(self):

        if not math.isclose(0.0, self.N, abs_tol=self.atol, rel_tol=self.rtol):
            return self.mod_M / self.N

        if math.isclose(0.0, self.mod_M, abs_tol=self.atol, rel_tol=self.rtol):
            raise ArithmeticError("No se puede determinar una excentricidad válida.")

        return np.inf

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
        return f"N = {self.N:8.1f} - My = {self.My:7.1f} - Mz = {self.Mz:7.1f}"

    def __eq__(self, other):
        if not isinstance(other, Force):
            return NotImplemented

        return (
                math.isclose(self.N, other.N, abs_tol=self.atol, rel_tol=self.rtol) and
                math.isclose(self.My, other.My, abs_tol=self.atol, rel_tol=self.rtol) and
                math.isclose(self.Mz, other.Mz, abs_tol=self.atol, rel_tol=self.rtol)
        )

    def __mul__(self, factor):
        if isinstance(factor, (int, float)):
            return Force(self.N * factor, self.My * factor, self.Mz * factor)
        raise TypeError("El multiplicador debe ser un escalar (int o float)")

    def __rmul__(self, factor):
        return self.__mul__(factor)


class StrainPlane:

    def __init__(self, theta=0, kappa=0, xo=0):
        """
            Define un plano de deformaciones que puede desplazarse sobre el eje "x" que define el eje de la barra,
            girar alrededor del mismo un ángulo theta entre 0 y 2 pi, e inclinarse girando sobre un eje horizontal
            paralelo al eje neutro de la sección.

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
        @param point: Un punto en dos o tres dimensiones.
        @return: La distancia entre el punto y el eje baricéntrico paralelo al neutro de la sección, en mm.
        """

        if isinstance(point, Point2D):
            point = [0, point.y, point.z]

        return np.cross(point, self.nn)[0]

    def get_dist_nn(self, point):
        """
            Calcula la distancia de que hay entre el punto y el eje neutro de la sección, o el eje baricéntrico en caso
            de que el plano sea horizontal. La distancia se toma positiva si la fibra se encuentra del lado del eje
            neutro donde una curvatura positiva genera compresión.

        @param point: Un punto en dos o tres dimensiones.
        @return: La distancia entre el punto y el eje neutro de la sección, en mm.
        """
        if isinstance(point, Point2D):
            point = [0, point.y, point.z]

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
        return f"theta = {self.theta:3.2f} rad - \u03BA = {100 * self.kappa:.3f} % - xo = {1000 * self.xo:.3f} \u2030"


class Stirrups:

    def __init__(self, stirrup_type=1, number=None, diam=None, sep=None):
        super().__init__()
        self.stirrup_type = stirrup_type
        self.number = number
        self.diam = diam
        self.sep = sep
