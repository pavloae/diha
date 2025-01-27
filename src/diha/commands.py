import logging
import os

import click
import numpy as np

from diha.builders import SectionBuilder
from diha.components import Force
from diha.plotter import plot_diagram_2d, plot_section, plot_tension, plot_diagram_3d
from diha.utils import norm_ang


def get_logger(level):
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    return logger


@click.group()
def cli():
    pass

@cli.command()
@click.argument('section_file', type=click.Path(exists=True))
@click.option('--output_file', default=None, help='Nombre del archivo de salida con el grafico de la sección')
@click.option('--debug', is_flag=True, show_default=False)
def section(section_file, **kwargs):
    """
    diha section Commandline

    Grafica una sección predefinida.
    """

    output_file = kwargs.get('output_file')
    if not output_file:
        output_file = os.path.splitext(section_file)[0] + '.svg'

    debug = kwargs.pop('debug')
    get_logger(logging.DEBUG if debug else logging.INFO)

    section = SectionBuilder().from_json(section_file)
    plot_section(section, file=output_file)


@cli.command()
@click.argument('section_file', type=click.Path(exists=True))
@click.option('--output_file', default=None, help='Nombre del archivo de salida con el grafico del diagrama')
@click.option('--theta', required=False, type=float, show_default=True, default=0, help='Rotación del vector de momentos en grados respecto del eje +Z')
@click.option('--points', required=False, type=int, show_default=True, default=32, help="Numero de puntos a calcular sobre la curva de interacción")
@click.option('--debug', is_flag=True, show_default=False)
def diagram(section_file, **kwargs):
    """
    diha diagram Commandline

    Grafica un diagrama de interacción en 2D para una sección predefinida.
    """

    output_file = kwargs.get('output_file')
    if not output_file:
        output_file = os.path.splitext(section_file)[0] + '.svg'

    debug = kwargs.pop('debug')
    get_logger(logging.DEBUG if debug else logging.INFO)

    theta = norm_ang(kwargs.pop('theta', 0) * np.pi / 180)

    section = SectionBuilder().from_json(section_file)
    plot_diagram_2d(section, theta, points=kwargs.get('points', 32), file=output_file)


@cli.command()
@click.argument('section_file', type=click.Path(exists=True))
@click.option('--output_file', default=None, help='Nombre del archivo de salida con el grafico del diagrama')
@click.option('--force', '-F', nargs=3, type=float, default=(0.0, 0.0, 0.0),
              help='Esfuerzo axil (kN), esfuerzos flectores (My, Mz) como valores separados por coma.')
@click.option('--debug', is_flag=True, show_default=False, help='Habilita el modo de depuración.')
def tension(section_file, force, debug, **kwargs):
    """
    Grafica las tensiones de la fuerza nominal correspondiente a la fuerza especificada.
    """

    output_file = kwargs.get('output_file')
    if not output_file:
        output_file = os.path.splitext(section_file)[0] + '.svg'

    logging_level = logging.DEBUG if debug else logging.INFO
    get_logger(logging_level)

    N, My, Mz = force
    force = Force(N=N*1e3, My=My*1e6, Mz=Mz*1e6)

    section = SectionBuilder().from_json(section_file)
    section.get_nominal_force(force)
    plot_tension(section, file=output_file)


@cli.command()
@click.argument('section_file', type=click.Path(exists=True))
@click.option('--output_file', default=None, help='Nombre del archivo de salida con el grafico del diagrama')
@click.option('--points', required=False, type=int, show_default=True, default=32, help="Numero de puntos a calcular sobre la curva de interacción")
@click.option('--debug', is_flag=True, show_default=False)
def diagram3d(section_file, **kwargs):
    """
    diha diagram Commandline

    Grafica un diagrama de interacción en 2D para una sección predefinida.
    """

    output_file = kwargs.get('output_file')
    if not output_file:
        output_file = os.path.splitext(section_file)[0] + '.html'

    debug = kwargs.pop('debug')
    get_logger(logging.DEBUG if debug else logging.INFO)

    section = SectionBuilder().from_json(section_file)
    plot_diagram_3d(section, points=kwargs.get('points', 32), file=output_file)


@cli.command()
@click.argument('section_file', type=click.Path(exists=True))
@click.option('--force', '-F', nargs=3, type=float, default=(0.0, 0.0, 0.0),
              help='Esfuerzo axil (kN), esfuerzos flectores (My, Mz) como valores separados por coma.')
@click.option('--debug', is_flag=True, show_default=False, help='Habilita el modo de depuración.')
def rel(section_file, force, debug):
    """
    Obtiene la relación entre la resistencia requerida y la resistencia de diseño para una sección predefinida.
    """
    logging_level = logging.DEBUG if debug else logging.INFO
    get_logger(logging_level)

    N, My, Mz = force
    Fu = Force(N=N*1e3, My=My*1e6, Mz=Mz*1e6)

    section = SectionBuilder().from_json(section_file)
    result = section.get_rel(Fu)

    click.echo(result)
