from unittest import TestCase

import numpy as np

from diha.plotter import plot_section, plot_tension, plot_diagram_2d
from section_factory import get_section_1


class Test(TestCase):

    def test_plot_section(self):
        section = get_section_1()
        plot_section(section)

    def test_plot_tension(self):
        section = get_section_1()
        section.set_limit_plane_by_strains(-0.003, 0.005, 0.25 * np.pi)
        plot_tension(section)

    def test_plot_diagram_2d(self):
        section = get_section_1()
        plot_section(section)
        plot_diagram_2d(section, 0)
