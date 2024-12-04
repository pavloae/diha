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
