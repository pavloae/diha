from diha.components import Stirrups
from diha.fibers import RoundFiber
from diha.materials import ConcreteMaterial, SteelMaterial
from diha.sections import RectangularRCSection

def get_section():
    concrete = ConcreteMaterial()
    steel = SteelMaterial()

    b = 500
    h = 750
    rec = 75

    diam = 25

    bars = [
        RoundFiber(steel, (h / 2 - rec, b / 2 - rec), diam), RoundFiber(steel, (h / 2 - rec, 0), diam), RoundFiber(steel, (h / 2 - rec, -b / 2 + rec), diam),
        RoundFiber(steel, (1/3*(h / 2 - rec), b / 2 - rec), diam), RoundFiber(steel, (1/3*(h / 2 - rec), -b / 2 + rec), diam),
        RoundFiber(steel, (-1/3*(h / 2 - rec), b / 2 - rec), diam), RoundFiber(steel, (-1/3*(h / 2 - rec), -b / 2 + rec), diam),
        RoundFiber(steel, (-h / 2 + rec, b / 2 - rec), diam), RoundFiber(steel, (-h / 2 + rec, 0), diam), RoundFiber(steel, (-h / 2 + rec, -b / 2 + rec), diam),
    ]

    section = RectangularRCSection(concrete, steel, stirrups=Stirrups(), b=b, h=h, bars=bars)

    return section