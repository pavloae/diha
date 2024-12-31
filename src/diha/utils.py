import numpy as np


def calc_angle_yz(u, v):
    """
        Calcula el ángulo en el plano yz para ir de la proyección del vector u a la proyección del vector "v" considerando
        positivo el giro en sentido antihorario (regla de la mano derecha)

    @param u: vector inicial
    @param v: vector final
    @return: un ángulo en radianes entre [0, 2*pi) con su signo

    Examples:
        >>> calc_angle_yz([1, 1, 0], [1, 1, 0])  # Vectores paralelos mismo sentido
        0.0
        >>> calc_angle_yz([1, 1, 0], [1, -1, 0])  # Vectores paralelos sentidos opuestos
        3.141592653589793
        >>> calc_angle_yz([1, 1, 0], [1, 0, 1])  # Giro antihorario 90°
        1.5707963267948966
        >>> calc_angle_yz([1, 1, 0], [1, 0, -1]) # Giro horario 270°
        4.71238898038469
        >>> calc_angle_yz([0, 0, 0], [1, 1, 0]) # Vector nulo como u
        0.0
        >>> calc_angle_yz([1, 1, 0], [0, 0, 0]) # Vector nulo como v
        0.0
    """

    up = np.array([0, *u[1:]])
    vp = np.array([0, *v[1:]])

    norm_u = np.linalg.norm(up)
    norm_v = np.linalg.norm(vp)

    if norm_u == 0 or norm_v == 0:
        return 0.0

    un = up / norm_u
    vn = vp / norm_v

    theta = np.arccos(np.dot(un, vn))

    return float(theta if np.cross(un, vn)[0] >= 0 else 2 * np.pi - theta)
