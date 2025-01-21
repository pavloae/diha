import os
from unittest import TestCase

from diha.builders import SectionBuilder
from diha.sections import RectangularRCSectionBase


class TestReinforcementConcreteSection(TestCase):

    def test_builder(self):
        file = os.path.join(os.path.dirname(__file__), 'files', 'section1.json')
        section = SectionBuilder().from_json(file)
        self.assertIsInstance(section, RectangularRCSectionBase)


