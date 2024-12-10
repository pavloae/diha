from unittest import TestCase

from test_sections import get_section


class TestReinforcementConcreteSection(TestCase):

    def test_plot(self):

        section = get_section()
        section.plot_section()
        section.plot_diagram()
