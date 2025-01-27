import numpy as np

from diha.components import Stirrups
from diha.fibers import RoundFiber
from diha.materials import ConcreteMaterial, SteelMaterial
from diha.sections import RectangularRCSectionBase


def get_section_1():
    concrete = ConcreteMaterial()
    steel = SteelMaterial()

    b = 500
    h = 750
    rec = 75

    diam = 25

    bars = [
        RoundFiber(steel, (h / 2 - rec, b / 2 - rec), diam), RoundFiber(steel, (h / 2 - rec, 0), diam),
        RoundFiber(steel, (h / 2 - rec, -b / 2 + rec), diam),
        RoundFiber(steel, (1 / 3 * (h / 2 - rec), b / 2 - rec), diam),
        RoundFiber(steel, (1 / 3 * (h / 2 - rec), -b / 2 + rec), diam),
        RoundFiber(steel, (-1 / 3 * (h / 2 - rec), b / 2 - rec), diam),
        RoundFiber(steel, (-1 / 3 * (h / 2 - rec), -b / 2 + rec), diam),
        RoundFiber(steel, (-h / 2 + rec, b / 2 - rec), diam), RoundFiber(steel, (-h / 2 + rec, 0), diam),
        RoundFiber(steel, (-h / 2 + rec, -b / 2 + rec), diam),
    ]

    section = RectangularRCSectionBase(concrete, steel, stirrups=Stirrups(), b=b, h=h, bars=bars)

    return section


def get_section_2():
    concrete = ConcreteMaterial()
    steel = SteelMaterial()

    b = 200
    h = 600
    rec = 30

    diam = 16

    bars = [
        RoundFiber(steel, (h / 2 - rec, b / 2 - rec), diam), RoundFiber(steel, (h / 2 - rec, 0), diam),
        RoundFiber(steel, (h / 2 - rec, -b / 2 + rec), diam),
        RoundFiber(steel, (-h / 2 + rec, b / 2 - rec), diam), RoundFiber(steel, (-h / 2 + rec, 0), diam),
        RoundFiber(steel, (-h / 2 + rec, -b / 2 + rec), diam),
    ]

    section = RectangularRCSectionBase(concrete, steel, stirrups=Stirrups(), b=b, h=h, bars=bars)
    section.rec = rec

    return section


def get_section_3():
    concrete = ConcreteMaterial(fpc=25)
    steel = SteelMaterial(fy=420)
    b, h = 120, 400
    As = 443

    diam = np.sqrt((As / 4) * 4 / np.pi)
    rec = 20 + 6 + diam / 2
    y1 = - h / 2 + rec

    d = 347
    yg = h / 2 - d

    sep = 2 * (yg - y1)
    y2 = y1 + sep

    bars = [
        RoundFiber(steel, (y1, b / 2 - rec), diam), RoundFiber(steel, (y1, -b / 2 + rec), diam),
        RoundFiber(steel, (y2, b / 2 - rec), diam), RoundFiber(steel, (y2, -b / 2 + rec), diam),
    ]
    return RectangularRCSectionBase(concrete, steel, b, h, bars)


def get_section_4():
    concrete = ConcreteMaterial()
    steel = SteelMaterial()

    b = 200
    h = 200
    rec = 25

    diam = 12

    bars = [
        RoundFiber(steel, (h / 2 - rec, b / 2 - rec), diam), RoundFiber(steel, (h / 2 - rec, -b / 2 + rec), diam),
        RoundFiber(steel, (-h / 2 + rec, b / 2 - rec), diam), RoundFiber(steel, (-h / 2 + rec, -b / 2 + rec), diam),
    ]

    section = RectangularRCSectionBase(concrete, steel, stirrups=Stirrups(), b=b, h=h, bars=bars)

    return section
