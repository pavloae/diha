import math

import numpy as np

from .utils import rotation_matrix


class Force:

    tol = 1e-6  # Tolerancia

    def __init__(self, N=0, My=0, Mz=0):
        self.N = N   # Fuerza normal
        self.My = My # Momento respecto al eje y
        self.Mz = Mz # Momento respecto al eje z

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
            ey = 0 if self.Mz == 0 else -np.inf
            ez = 0 if self.My == 0 else np.inf

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


class StrainPlane:

    def __init__(self, n=None, epsilon_o=None):
        super().__init__()

        # Vector unitario normal al plano
        self._n = None

        # Vector unitario que define la dirección del eje neutro
        self._nn = None

        # Vector que define el punto de intersección del plano con el eje x
        self._nx = np.array([0, 0, 0])

        self.epsilon_o = None

        self.kappa_y = 0
        self.kappa_z = 0

        self.n = n or [1, 0, 0]
        self.set_epsilon_o(epsilon_o or 0)

    @property
    def n(self):

        if not any(self._n) or self._n[0] <= 0:
            raise ValueError("El vector normal al plano de deformación no está definido correctamente")

        return self._n

    @n.setter
    def n(self, n):

        if not any(n) or n[0] <= 0:
            raise ValueError("El vector normal al plano de deformación no está definido correctamente")

        n = np.array(n)
        self._n = n / np.linalg.norm(n)

        # Si el vector normal al plano conicide con el eje x se define el eje neutro en el sentido del eje z positivo
        if np.allclose(self._n, np.array([1, 0, 0])):
            self._nn = np.array([0, 0, 1])
        else:
            nn = np.cross([1, 0, 0], self.n)
            self._nn = nn / np.linalg.norm(nn)

        self._set_kappa_y()
        self._set_kappa_z()

    def set_nx(self, nx):
        self._nx = nx
        self.epsilon_o = self._nx[0]

    def set_epsilon_o(self, epsilon_o):
        self.epsilon_o = epsilon_o
        self._nx = np.array([self.epsilon_o, 0, 0])

    def _set_kappa_z(self):
        k_xy = np.cross([0, 0, 1], self.n)
        self.kappa_z = k_xy[0] / k_xy[1]

    def _set_kappa_y(self):
        k_zx = np.cross([0, 1, 0], self.n)
        self.kappa_y = k_zx[0] / k_zx[2]

    def rotate(self, theta):
        """
            Rota el plano de deformaciones alrededor del eje x un ángulo theta. El signo del ángulo se corresponde con
            la regla de la mano derecha.
        :param theta: Ángulo de rotación [rad].
        """
        rotator = rotation_matrix(np.array([1, 0, 0]), theta)
        self.n = np.dot(rotator, self.n)

    def inclinate(self, theta, axis='nn'):
        """
            Rota el plano de deformaciones alrededor del eje y, z o del eje nn (eje neutro) un ángulo theta para
            inclinarlo.

        :param theta: Ángulo de rotación [rad].
        :param axis: El eje sobre el cual va a rotar el plano: x, y o nn (default).
        """

        if axis == 'y':
            v = np.array([0, 1, 0])
        elif axis == 'z':
            v = np.array([0, 0, 1])
        else:
            v = self._nn

        rotator = rotation_matrix(v, theta)
        self.n = np.dot(rotator, self.n)

    def move(self, epsilon_x):
        self.set_nx(self._nx + np.array([epsilon_x, 0, 0]))

    def get_strain(self, y, z):
        return self.epsilon_o + self.kappa_y * z - self.kappa_z * y

    def set_plain(self, point1, point2, point3):

        v1 = np.array(point1)
        v2 = np.array(point2)
        v3 = np.array(point3)

        n = np.cross(v3 - v1, v3 - v2)
        if n[0] < 0:
            n = -n

        self.n = n / np.linalg.norm(n)

        epsilon_o = point1[0] - self.kappa_y * point1[2] + self.kappa_z * point1[1]
        self.set_nx(np.array([epsilon_o, 0, 0]))

    def __str__(self):
        return f"eo={1000*self.epsilon_o:8.2f}\u2030 - \u03BAz ={1000*self.kappa_y:10.2f} - \u03BAy ={1000*self.kappa_y:10.3f}"
