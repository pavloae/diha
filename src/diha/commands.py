import logging
import os

import click
import numpy as np

from diha.builders import SectionBuilder
from diha.utils import norm_ang


def get_logger(level):
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    return logger


@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output_file', default=None, help='Nombre del archivo de salida con el grafico del diagrama')
@click.option('--theta', required=False, type=float, show_default=True, default=0, help='Rotación del vector de momentos en grados respecto del eje +Z')
@click.option('--points', required=False, type=int, show_default=True, default=32, help="Numero de puntos a calcular sobre la curva de interacción")
@click.option('--debug', is_flag=True, show_default=False)
def plot(input_file, **kwargs):
    """
    diha plot Commandline

    Grafica un diagrama de interacción en 2D para una sección predefinida.
    """

    output_file = kwargs.get('output_file')
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.svg'

    debug = kwargs.pop('debug')
    get_logger(logging.DEBUG if debug else logging.INFO)

    theta = norm_ang(kwargs.pop('theta', 0) * np.pi / 180)

    section = SectionBuilder().from_json(input_file)
    section.plot_diagram_2d(theta_me=theta, points=kwargs.get('points', 32), file=output_file)
