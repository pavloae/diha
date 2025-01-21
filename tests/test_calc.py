import pickle
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from diha.components import Force, StrainPlane, Stirrups
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

class TestReinforcementConcreteSection(TestCase):

    def test_serializable(self):
        section = get_section_1()
        data = pickle.dumps(section)
        self.assertIsNotNone(data)

        serialized_object = pickle.loads(data)
        self.assertIsNotNone(serialized_object)

    def test_get_force_by_strain_limit(self):
        section = get_section_1()

        Pnc = 0.85 * section.concrete.fpc * (section.Ag - section.As)
        Pns = section.steel.fy * section.As
        esy = section.steel.fy / section.steel.E

        Pn = Pnc + Pns

        # Mientras la fibra menos comprimida del acero esté en fluencia las fuerzas internas deberían ser iguales
        section.set_limit_plane_by_strains(-0.003, -0.003, 0)
        self.assertAlmostEqual(-Pn, section.force_i.N, delta=1e3)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertAlmostEqual(0, section.force_i.Mz)

        section.set_limit_plane_by_strains(-0.003, -esy, 0)
        self.assertAlmostEqual(-Pn, section.force_i.N, delta=1e3)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertAlmostEqual(0, section.force_i.Mz)

        # Cuando la fibra menos comprimida del acero entra en el periodo elástico (ϵ<fy/E) comienzan a variar las
        # fuerzas internas
        epsilon_y = section.steel.fy / section.steel.E
        section.set_limit_plane_by_strains(-0.003, -0.95 * epsilon_y, 0)
        self.assertGreaterEqual(section.force_i.N, -Pn)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertGreaterEqual(section.force_i.Mz, 0)

        section.set_limit_plane_by_strains(-0.003, 0.005, 0)
        self.assertGreaterEqual(section.force_i.N, -Pn)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertGreaterEqual(section.force_i.Mz, 0)

        # ... Y siguen variando hasta que la fibra menos traccionada del acero entre en fluencia
        top_concrete = np.max([fiber.center[0] for fiber in section.concrete_fibers])
        top_steel = np.max([fiber.center[0] for fiber in section.steel_fibers])
        bottom_steel = np.min([fiber.center[0] for fiber in section.steel_fibers])
        epsilon_c = epsilon_y / (top_steel - bottom_steel) * (top_concrete - bottom_steel)

        section.set_limit_plane_by_strains(0.95 * epsilon_c, 0.005, 0)
        self.assertGreaterEqual(section.force_i.N, -Pn)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertGreaterEqual(section.force_i.Mz, 0)

        # Cuando la fibra más traccionada del acero alcanza el límite plástico las fuerza internas vuelven a mantenerse
        # iguales.
        section.set_limit_plane_by_strains(epsilon_c, 0.005, 0)
        self.assertAlmostEqual(Pns, section.force_i.N, delta=1e3)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertAlmostEqual(0, section.force_i.Mz)

        section.set_limit_plane_by_strains(0.005, 0.005, 0)
        self.assertAlmostEqual(Pns, section.force_i.N, delta=1e3)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertAlmostEqual(0, section.force_i.Mz)

    def test_set_limit_plane_by_strains(self):
        # Sección con armadura simétrica
        section = get_section_1()

        # Flexión positiva alrededor del eje "y"
        section.set_limit_plane_by_strains(-0.003, 0.005, 1.5 * np.pi)
        self.assertGreaterEqual(section.force_i.My, 0.00)
        self.assertAlmostEqual(0.00, section.force_i.Mz, places=5)

        # Compresión pura
        section.set_limit_plane_by_strains(-0.003, -0.003, 0)
        self.assertLessEqual(section.force_i.N, 0)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertAlmostEqual(0, section.force_i.Mz)

        # Tracción pura
        section.set_limit_plane_by_strains(0.005, 0.005, 0)
        self.assertGreaterEqual(section.force_i.N, 0)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertAlmostEqual(0, section.force_i.Mz)

        # Flexión positiva alrededor del eje "z"
        section.set_limit_plane_by_strains(-0.003, 0.005, 0)
        self.assertAlmostEqual(0, section.force_i.My)
        self.assertGreaterEqual(section.force_i.Mz, 0)

    def test_set_limit_plane_by_eccentricity(self):
        section = get_section_1()

        self.assertRaises(ValueError, section.set_limit_plane_by_eccentricity, 0, 0)

    def test_get_forces(self):
        section = get_section_1()
        forces = section.get_forces()

        self.assertIsNotNone(forces)

    def test_plot_section(self):
        section = get_section_1()
        section.plot_section()

    def test_plot_tension(self):
        section = get_section_1()
        section.set_limit_plane_by_strains(-0.003, 0.005, 0.25 * np.pi)
        section.plot_tension()

    def test_plot_diagram_2d(self):
        section = get_section_1()
        section.plot_section()
        section.plot_diagram_2d(theta_me=0)

    def test_get_limit_strains(self):
        section = get_section_1()
        self.assertAlmostEqual((-0.003, -0.003), section.get_limits_strain(0))
        self.assertAlmostEqual((-0.003, 0.001), section.get_limits_strain(.25))
        self.assertAlmostEqual((-0.003, 0.005), section.get_limits_strain(.5))
        self.assertAlmostEqual((0.001, 0.005), section.get_limits_strain(.75))
        self.assertAlmostEqual((0.005, 0.005), section.get_limits_strain(1))

    def test_get_farthest_fiber(self):
        section = get_section_2()
        section.build()

        # Flexión alrededor del eje z
        strain_plane = StrainPlane(0, 1, 0)
        delta_y = section.h / section.div_y

        ffc = section.get_farthest_fiber(section.concrete_fibers, strain_plane, compression=True)
        dist = section.h / 2 - delta_y / 2
        self.assertAlmostEqual(ffc.center.y, dist)

        ffs = section.get_farthest_fiber(section.steel_fibers, strain_plane, compression=False)
        dist = section.steel_fibers[-1].y
        self.assertAlmostEqual(ffs.center.y, dist)

        # Flexión alrededor del eje y
        strain_plane = StrainPlane(1.5 * np.pi, 1, 0)
        delta_z = section.b / section.div_z

        ffc = section.get_farthest_fiber(section.concrete_fibers, strain_plane, compression=True)
        dist = section.b / 2 - delta_z / 2
        self.assertAlmostEqual(ffc.center.z, -dist)

        ffs = section.get_farthest_fiber(section.steel_fibers, strain_plane, compression=False)
        dist = section.steel_fibers[0].z
        self.assertAlmostEqual(ffs.center.z, dist)

    def test_get_params(self):
        section = get_section_1()
        section.build()

        kappa, xo = section._get_params(-0.003, -0.003)
        self.assertAlmostEqual(kappa, 0)
        self.assertAlmostEqual(xo, -0.003)

        kappa, xo = section._get_params(0.005, 0.005)
        self.assertAlmostEqual(kappa, 0)
        self.assertAlmostEqual(xo, 0.005)

        assert isinstance(section, RectangularRCSectionBase)
        section.div_y *= 5
        section.div_z *= 5
        section.build(force=True)
        rec = .5 * (section.h - 2 * section.steel_fibers[0].center[0])
        d = section.h - rec

        ec, es = -0.003, 0.0
        kappa, xo = section._get_params(ec, es)
        self.assertAlmostEqual(kappa, -ec / d)
        self.assertAlmostEqual(xo, ec + kappa * section.h / 2, places=4)

        ec, es = -0.003, 0.005
        kappa, xo = section._get_params(ec, es)
        self.assertAlmostEqual(kappa, (es - ec) / d)
        self.assertAlmostEqual(xo, ec + kappa * section.h / 2, places=4)

        section.div_y *= 5
        section.div_z *= 5
        section.build(force=True)
        section.strain_plane = StrainPlane(theta=1.5 * np.pi)
        d = section.b - rec
        ec, es = -0.003, 0.005
        kappa, xo = section._get_params(ec, es)
        self.assertAlmostEqual(kappa, (es - ec) / d)
        self.assertAlmostEqual(xo, ec + kappa * section.b / 2, places=4)

    def test_get_params_2(self):
        section = get_section_1()

        section.div_y *= 2
        section.div_z *= 2
        section.build()

        section.strain_plane = StrainPlane(theta=1.5 * np.pi)
        rec = .5 * section.b - section.steel_fibers[0].center.z
        delta_z = section.b / section.div_z
        d_calc = section.b - rec - delta_z / 2
        h_calc = section.b - delta_z
        ec, es = -0.003, 0.005
        kappa, xo = section._get_params(ec, es)
        self.assertAlmostEqual(kappa, (es - ec) / d_calc)
        self.assertAlmostEqual(xo, ec + kappa * h_calc / 2)

    def test_set_limit_plane(self):
        section = get_section_1()
        section.build()

        section.set_limit_plane_by_strains(-0.003, -0.003, theta_me=0)

    def test_design_force(self):
        section = get_section_1()

        self.assertAlmostEqual(section.An, section.Ag - section.As)

        # Tracción pura

        Pn = section.steel.fy * section.As
        Pd = 0.90 * Pn

        section.get_nominal_force(0, 0, compression=False)
        self.assertAlmostEqual(section.get_nominal_force(0, 0, compression=False).N, Pn)
        self.assertAlmostEqual(section.get_design_force(0, 0, compression=False).N, Pd)

        # Compresión pura

        Pn = -(0.85 * section.concrete.fpc * section.An + section.steel.fy * section.As)
        Pd = 0.65 * 0.80 * Pn

        section.force_e = Force(N=-10)
        self.assertAlmostEqual(section.get_nominal_force(0, 0, compression=True).N, Pn)
        self.assertAlmostEqual(section.get_design_force(0, 0, compression=True).N, Pd)

        # Flexo-compresión

        section.force_e = Force(N=-2000e3, Mz=200e6)
        self.assertAlmostEqual(-3900e3, section.get_design_force(section.force_e.e, 0).N, delta=100e3)

    def test_bending_sample_2_I_1(self):

        # Datos del problema
        Mu = 52.0

        concrete = ConcreteMaterial(fpc=25)
        steel = SteelMaterial(fy=420)
        b, h = 120, 400
        As = 443

        diam = np.sqrt((As / 4) * 4 / np.pi)
        rec = 20 + 6 + diam / 2

        y1 = - h / 2 + rec  # Coordenada de la capa inferior
        d = 347  # Altura de calculo utilizada
        yg = h / 2 - d  # Coordenadas del baricentro de las armaduras
        y2 = y1 + 2 * (yg - y1)  # Coordenadas de la capa superior

        bars = [
            RoundFiber(steel, (y1, b / 2 - rec), diam), RoundFiber(steel, (y1, -b / 2 + rec), diam),
            RoundFiber(steel, (y2, b / 2 - rec), diam), RoundFiber(steel, (y2, -b / 2 + rec), diam),
        ]
        section = RectangularRCSectionBase(concrete, steel, b, h, bars)

        # Aproximación por cargas de compresión
        ee = -np.inf
        section.set_limit_plane_by_eccentricity(ee, 0)
        self.assertAlmostEqual(Mu, section.get_design_force(ee, 0).Mz * 1e-6, delta=0.01 * Mu)

        # Aproximación por cargas de tracción
        ee = np.inf
        section.set_limit_plane_by_eccentricity(ee, 0)
        self.assertAlmostEqual(Mu, section.get_design_force(ee, 0).Mz * 1e-6, delta=0.01 * Mu)

    def test_bending_sample_2_I_7(self):

        # Datos del problema
        concrete = ConcreteMaterial(fpc=25)
        steel = SteelMaterial(fy=420)
        b, h = 120, 400
        diam = 16
        rec = 20 + 6 + 16 / 2

        y1 = h / 2 - rec  # Coordenada de la capa de compresión
        y2 = - h / 2 + rec  # Coordenadas de la capa superior

        bars = [
            RoundFiber(steel, (y1, b / 2 - rec), diam), RoundFiber(steel, (y1, -b / 2 + rec), diam),
            RoundFiber(steel, (y2, b / 2 - rec), diam), RoundFiber(steel, (y2, -b / 2 + rec), diam),
        ]
        section = RectangularRCSectionBase(concrete, steel, b, h, bars)

        # Aproximación por cargas de compresión
        ee = -np.inf
        section.set_limit_plane_by_eccentricity(ee, 0)
        self.assertAlmostEqual(51.66, section.get_design_force(ee, 0).Mz * 1e-6, delta=0.3)

    def test_spp_function(self):
        section = get_section_1()
        section.increase_resolution(2)

        section.set_limit_plane_by_strains(-0.003, -0.00284, theta_me=0)
        section.analyze()

        n = 100
        spp = np.arange(0, n + 1) / n
        strains = [section.get_limits_strain(param) for param in spp]

        eccentricity = []
        for strain in strains:
            section.set_limit_plane_by_strains(*strain, theta_me=np.pi)
            eccentricity.append(section.force_i.e)

        plt.plot(spp, eccentricity)
        plt.show()

    def test_analyze(self):
        section = get_section_3()
        section.set_limit_plane_by_strains(-0.003, -0.003, theta_me=0)
        section.analyze()
