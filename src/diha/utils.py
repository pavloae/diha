import math

import numpy as np


def calc_angle_yz(u, v):
    """
        Calcula el ángulo en el plano yz para ir de la proyección del vector u a la proyección del vector "v" considerando
        positivo el giro en sentido antihorario (regla de la mano derecha)

    @param u: vector inicial
    @param v: vector final
    @return: un ángulo en entre [0, 2*pi), en radianes

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
        Traceback (most recent call last):
        ...
        ValueError: Los vectores deben tener una componente en el plano yz no nula
        >>> calc_angle_yz([1, 1, 0], [0, 0, 0]) # Vector nulo como v
        Traceback (most recent call last):
        ...
        ValueError: Los vectores deben tener una componente en el plano yz no nula
    """

    up = np.array([0, *u[1:]])
    vp = np.array([0, *v[1:]])

    norm_u = np.linalg.norm(up)
    norm_v = np.linalg.norm(vp)

    if norm_u == 0 or norm_v == 0:
        raise ValueError('Los vectores deben tener una componente en el plano yz no nula')

    un = up / norm_u
    vn = vp / norm_v

    theta = np.arccos(np.dot(un, vn))

    return float(theta if np.cross(un, vn)[0] >= 0 else 2 * np.pi - theta)

def norm_ang(ang):
    """
        Normaliza un ángulo para que se encuentre en el rango [0, 2 pi)
    @param ang: Ángulo a ser normalizado, en radianes
    @return: Ángulo normalizado, en radianes
    """
    return ang % (2 * np.pi)

def is_same_direction(u, v):
    """
        Determina si dos vectores tienen la misma dirección.
    @param u: Un vector inicial
    @param v: Un vector final
    @return: Un booleano: True si los vectores forman un ángulo de 0 o pi radianes, False en caso contrario.
    """
    result = calc_angle_yz(u, v) % np.pi
    return math.isclose(0.0, result, abs_tol=1e-6) or math.isclose(result, np.pi, abs_tol=1e-6)