import numpy as np


def calc_angle_yz(u, v):
    """
        Calcula el ángulo en el plano yz para ir de la proyección del vector u a la proyección del vector v considerando
        positivo el giro en sentido antihorario (regla de la mano derecha)

    @param u: vector inicial
    @param v: vector final
    @return: un ángulo en radianes entre [0, 2*pi) con su signo
    """

    up = np.array([0, *u[1:]])
    vp = np.array([0, *v[1:]])

    norm_u = np.linalg.norm(up)
    norm_v = np.linalg.norm(vp)

    if norm_u == 0 or norm_v == 0:
        return 0

    un = up / norm_u
    vn = vp / norm_v

    theta = np.arccos(np.dot(un, vn))

    return theta if np.cross(un, vn)[0] >= 0 else 2 * np.pi - theta
