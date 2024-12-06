import numpy as np


def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)  # Normaliza el eje
    ux, uy, uz = axis
    cos = np.cos(theta)
    sin = np.sin(theta)

    # Construye la matriz de rotación
    R = np.array([
        [cos + ux ** 2 * (1 - cos), ux * uy * (1 - cos) - uz * sin, ux * uz * (1 - cos) + uy * sin],
        [uy * ux * (1 - cos) + uz * sin, cos + uy ** 2 * (1 - cos), uy * uz * (1 - cos) - ux * sin],
        [uz * ux * (1 - cos) - uy * sin, uz * uy * (1 - cos) + ux * sin, cos + uz ** 2 * (1 - cos)]
    ])
    return R


def normalize(v):
    """
        Normaliza un vector
    @param v: Vector
    @return:Vector normalizado
    """
    return v / np.linalg.norm(v)


def angle(a, b):
    """
        Calcula el ángulo desde el vector a al vector b

    :param a: El vector inicial
    :param b: El vector final
    :return: El ángulo entre los vectores, comprendido entre 0 y 2 pi [rad]
    """
    # Calcula el ángulo usando arctan2
    theta = np.arctan2(a, b)

    # Ajusta el rango a [0, 2*pi)
    if theta < 0:
        theta += 2 * np.pi

    return theta

def calc_angle_yz(u, v):
    """
        Calcula el ángulo en el plano yz para ir de la proyección del vector u a la proyección del vector v considerando
        positivo el giro en sentido antihorario (regla de la mano derecha)

    @param u: vector inicial
    @param v: vector final
    @return: un ángulo en radianes entre [0, 2*pi) con su signo
    """

    u[0] = 0
    v[0] = 0

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        return 0

    un = u / norm_u
    vn = v / norm_v

    theta = np.arccos(np.dot(un, vn))

    return theta if np.cross(un, vn)[0] >= 0 else 2 * np.pi - theta
