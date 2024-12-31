import pickle
from unittest import TestCase

import numpy as np

from diha.sections import RectangularRCSection
from test_sections import get_section


class TestReinforcementConcreteSection(TestCase):

    def test_serializable(self):
        section = get_section()
        data = pickle.dumps(section)
        self.assertIsNotNone(data)

        serialized_object = pickle.loads(data)
        self.assertIsNotNone(serialized_object)

    def test_get_force_by_strain_limit(self):
        section = get_section()

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

    def test_set_limit_plane_by_eccentricity(self):
        section = get_section()

        self.assertRaises(ValueError, section.set_limit_plane_by_eccentricity, 0, 0)

    def test_get_forces(self):
        section = get_section()
        forces = section.get_forces()

        self.assertIsNotNone(forces)

    def test_plot_section(self):
        section = get_section()
        section.plot_section()

    def test_plot_diagram_2d(self):
        section = get_section()
        section.plot_diagram_2d(theta_me=0*3.14)

    def test_get_limit_strains(self):
        section = get_section()
        self.assertAlmostEqual((-0.003, -0.003), section.get_limits_strain(0))
        self.assertAlmostEqual((-0.003, 0.001), section.get_limits_strain(.25))
        self.assertAlmostEqual((-0.003, 0.005), section.get_limits_strain(.5))
        self.assertAlmostEqual((0.001, 0.005), section.get_limits_strain(.75))
        self.assertAlmostEqual((0.005, 0.005), section.get_limits_strain(1))

    def test_get_params(self):
        section = get_section()
        section.build()

        kappa, xo = section._get_params(-0.003, -0.003)
        self.assertAlmostEqual(kappa, 0)
        self.assertAlmostEqual(xo, -0.003)

        kappa, xo = section._get_params(0.005, 0.005)
        self.assertAlmostEqual(kappa, 0)
        self.assertAlmostEqual(xo, 0.005)

        assert isinstance(section, RectangularRCSection)
        section.div_y *= 5
        section.div_z *= 5
        section.build(force=True)
        yp = -section.steel_fibers[-1].center[0]
        d = section.h / 2 + yp

        ec, es = -0.003, 0.0
        kappa, xo = section._get_params(ec, es)
        self.assertAlmostEqual(kappa, -ec/d)
        self.assertAlmostEqual(xo, ec + kappa * section.h / 2, places=4)

        ec, es = -0.003, 0.005
        kappa, xo = section._get_params(ec, es)
        self.assertAlmostEqual(kappa, (es - ec) / d)
        self.assertAlmostEqual(xo, ec + kappa * section.h / 2, places=4)

    def test__set_limit_plane(self):
        section = get_section()
        section.build()

        section.set_limit_plane_by_strains(-0.003, -0.003, theta_me=0)



