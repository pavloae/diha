import logging
from unittest import TestCase

import numpy as np

from diha.components import Force
from src.diha.fibers import RoundFiber
from src.diha.materials import ConcreteMaterial, SteelMaterial
from src.diha.sections import RectangularRCSection

logging.basicConfig(level=logging.DEBUG)

class TestInteractionDiagram(TestCase):

    @staticmethod
    def get_section():
        concrete = ConcreteMaterial()
        steel = SteelMaterial()

        b = 500
        h = 750
        rec = 50

        diam = 16

        bars = [
            RoundFiber(steel, (h / 2 - rec, b / 2 - rec), diam), RoundFiber(steel, (h / 2 - rec, -b / 2 + rec), diam),
            RoundFiber(steel, (0, b / 2 - rec), diam), RoundFiber(steel, (0, -b / 2 + rec), diam),
            RoundFiber(steel, (-h / 2 + rec, b / 2 - rec), diam), RoundFiber(steel, (-h / 2 + rec, -b / 2 + rec), diam),
        ]

        section = RectangularRCSection(concrete, steel, b, h, bars)

        return section

    def test_find_Tu(self):

        section = self.get_section()

        bar = section.steel_fibers[0]
        steel = bar.material
        assert isinstance(steel, SteelMaterial)

        As = bar.area

        # Se determina el axil límite de fluencia
        Ny = 6 * steel.fy * As

        # Axil para comportamiento elásico
        Ne = 0.8 * Ny

        # Axil para comportamiento plástico
        Np = 1.2 * Ny

        section.force_e = Force(Ne, 0, 0)

        Fu = section.calc_Fu()
        strain_plane = section.strain_plane

        # El plano de deformación es paralelo al plano de la sección
        self.assertTrue(np.allclose(strain_plane.n, [1, 0, 0]))

        # La fuerza axial en cada barra es la misma
        for bar in section.steel_fibers:
            self.assertAlmostEqual(bar.force.N, 1/6 * Ne)

        # La tensión en cada barra es la misma
        for bar in section.steel_fibers:
            self.assertAlmostEqual(bar.stress, 1/6 * Ne * 1/As)

        # La deformación en cada barra es la misma
        for bar in section.steel_fibers:
            self.assertAlmostEqual(bar.strain, 1/6 * Ne * 1/As * 1/steel.E)

        section.force_e = Force(Np, 0, 0)
        strain_plane = section.iterate()
        self.assertIsNone(strain_plane)

    def test_find_Pu(self):

        section = self.get_section()
        concrete = section.concrete
        b, h = section.b, section.h
        assert isinstance(concrete, ConcreteMaterial)

        bar = section.steel_fibers[0]
        steel = bar.material
        assert isinstance(steel, SteelMaterial)

        As = bar.area

        # Se determina el límite de rotura
        Nu = -6 * steel.fy * As - 0.85 * concrete.fpc * (b * h - 6 * As)

        N1 = 6 * steel.get_stress(-.0004) * As + 0.85 * concrete.get_stress(-.0004) * (b * h - 6 * As)

        section.N_external = N1
        strain_plane = section.iterate()
        self.assertIsNotNone(strain_plane)
        self.assertAlmostEqual(strain_plane.epsilon_o, -0.000341)

        # Axil para comportamiento elásico
        Ne = 0.8 * Nu

        # Axil para comportamiento plástico
        Np = 1.2 * Nu

        section.N_external = Ne
        strain_plane = section.iterate()

        # El plano de deformación es paralelo al plano de la sección
        self.assertIsNotNone(strain_plane)
        self.assertTrue(np.allclose(strain_plane.n, [1, 0, 0]))

        section.N_external = Np
        strain_plane = section.iterate()
        self.assertIsNone(strain_plane)

    def test_find_plane_N_Mz(self):

        section = self.get_section()

        bar = section.steel_fibers[0]
        steel = bar.material
        assert isinstance(steel, SteelMaterial)

        Nu = -1e3
        Mz = 0

        section.N_external = Nu
        section.Mz_external = Mz
        # section.build()
        # section.strain_plane.n = [1, 0, 0]
        # section.strain_plane._nx = []
        strain_plane = section.iterate()

        self.assertIsNotNone(strain_plane)
