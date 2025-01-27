import logging

import numpy as np
from matplotlib import pyplot as plt, cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

from diha.calc import ReinforcementConcreteSectionBase


def plot_section(section: ReinforcementConcreteSectionBase, file=None):

    fig, ax = plt.subplots(figsize=(6, 8))

    section.build()

    # Dibuja elementos de hormigón
    for fiber in section.concrete_fibers:
        fiber.plot(ax, color='gray')

    # Dibuja armaduras
    for fiber in section.steel_fibers:
        fiber.plot(ax, color='blue')

    # Configura gráfico
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.gca().invert_xaxis()
    plt.title(f"{section.__class__.__name__}")
    plt.grid(False)
    plt.autoscale()

    if file:
        plt.savefig(file, format='svg')
    else:
        plt.show()


def plot_tension(section: ReinforcementConcreteSectionBase, file=None):

    fig, ax = plt.subplots(figsize=(6, 8))

    section.build()

    stress = [fibra.stress for fibra in section.concrete_fibers + section.steel_fibers]
    min_stress = min(stress)
    max_stress = max(stress)

    norm = mcolors.Normalize(vmin=min_stress, vmax=max_stress)
    cmap = plt.get_cmap('bwr')

    # Dibuja elementos de hormigón
    for fiber in section.concrete_fibers:
        if fiber.stress < 0:
            fiber.plot(ax, color='gray')
        else:
            fiber.plot(ax, color='white')

    # Dibuja armaduras
    for fiber in section.steel_fibers:
        color = cmap(norm(fiber.stress))
        fiber.plot(ax, color=color)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Tensión del acero (MPa)')
    cbar.ax.invert_yaxis()  # Invertir la barra de colores para que el rojo esté arriba

    # Configura gráfico
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.gca().invert_xaxis()
    plt.suptitle(f"{section.__class__.__name__}")
    plt.title(f"fc={section.concrete.min_stress} MPa")
    plt.grid(False)
    plt.autoscale()

    if file:
        plt.savefig(file, format='svg')
        logger.info("Gráfico guardado como {}".format(file))
    else:
        plt.show()


def plot_diagram_2d(section: ReinforcementConcreteSectionBase, theta, points=32, file=None):
    nominal = []
    design = []

    for val in range(points + 1):
        section.set_limit_plane_by_strains(*section._get_limits_strain(val / points), theta)

        M, N = np.linalg.norm(section.force_i.M) * 1e-6, section.force_i.N * 1e-3

        nominal.append([M, N])

        factor = section.phi()
        design.append([factor * M, max(section.get_Pd_max() * 1e-3, factor * N)])

    x, y = zip(*nominal)
    plt.plot(x, y, marker='', linestyle='-', color='g', label='Nn-Mn')

    x, y = zip(*design)
    plt.plot(x, y, marker='', linestyle='-', color='r', label='Nd-Md')

    plt.xlabel('M [kNm]')
    plt.ylabel('N [kN]')

    plt.gca().invert_yaxis()

    plt.title(f'Diagrama de interacción - \u03B8={np.degrees(theta)}°')
    plt.legend()
    plt.grid(True)
    plt.autoscale()

    if file:
        plt.savefig(file, format='svg')
        logger.info("Gráfico guardado como {}".format(file))
    else:
        plt.show()


def plot_diagram_3d(section: ReinforcementConcreteSectionBase, points=32, file=None):

    # TODO: Completar
    # Datos para el gráfico
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

    # Crear superficie 3D
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title="Diagrama de interacción Mn-Nn")

    if file:
        fig.write_html(file)
        logger.info("Gráfico guardado como {}".format(file))
    else:
        fig.show()