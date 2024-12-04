import logging
from typing import List

import numpy as np

from .components import Force, StrainPlane
from .fibers import RectFiber, RoundFiber, GroupFiberStatus
from .materials import SteelMaterial, ConcreteMaterial
from .plotter import PlotterMixin
from .utils import angle

logger = logging.getLogger()


class ConcreteStrainExceededError(Exception):
    """Excepción lanzada cuando las deformaciones en el hormigón exceden los límites permitidos."""
    def __init__(self, strain, limit, message=None):
        self.strain = strain
        self.limit = limit
        if message is None:
            message = f"Strain exceeded: {strain:.2f} (Limit: {limit:.2f} MPa)"
        super().__init__(message)


class SteelStrainExceededError(Exception):
    """Excepción lanzada cuando las deformaciones en el acero exceden los límites permitidos."""
    def __init__(self, strain, limit, message=None):
        self.strain = strain
        self.limit = limit
        if message is None:
            message = f"Strain exceeded: {strain:.2f} (Limit: {limit:.2f} MPa)"
        super().__init__(message)


class ReinforcementConcreteSection(PlotterMixin):

    def __init__(self, concrete, steel, bars, N=0, My=0, Mz=0, max_initial_strain=0.001, iterations=5):

        super().__init__()

        self.max_iterations= iterations

        self.strain_plane = StrainPlane()

        self.concrete: ConcreteMaterial = concrete
        self.steel: SteelMaterial = steel

        self.steel_fibers: List[RoundFiber] = bars
        self.concrete_fibers: List[RectFiber] = []

        self.max_initial_strain = max_initial_strain

        self.force_e = Force(N, My, Mz)
        self.force_i = Force()

        self.concrete_status = GroupFiberStatus()
        self.steel_status = GroupFiberStatus()

        self._built = False

    def build(self, force=False):
        if not self._built or force:
            self._build_concrete_fibers()
            self._built = True

    def _build_concrete_fibers(self):
        raise NotImplementedError

    def analyze(self):

        ConcreteMaterial.diagram_type = 2

        self.build()

        for fiber in self.steel_fibers:
            fiber.set_strain(self.strain_plane)

        for fiber in self.concrete_fibers:
            fiber.set_strain(self.strain_plane)

        for fiber in self.steel_fibers + self.concrete_fibers:
            fiber.set_force()

        self.concrete_status.update(self.concrete_fibers)
        self.steel_status.update(self.steel_fibers)

        self.force_i = self.concrete_status.force + self.steel_status.force

    def iterate_depht(self):

        def condition():
            return self.force_e.N == self.force_i.N

        iteration = 0

        while not condition() and iteration < self.max_iterations:

            iteration += 1

            if self.strain_plane.epsilon_o == 0:

                epsilon_i = 0.0001 if self.force_e.N > 0 else -0.0001

            else:

                epsilon_i = self.force_e.N / self.force_i.N * self.strain_plane.epsilon_o

            self.strain_plane.set_epsilon_o(epsilon_i)
            self.analyze()


        self.report()

    def iterate_angle(self):

        def diff_angle():
            ve = self.force_e.e[1:]
            vi = self.force_i.e[1:]

            ang_e = angle(*ve)
            ang_i = angle(*vi)

            diff = ang_i - ang_e

            return diff

        print("============= Acomodando ángulo =================")

        def condition():
            return abs(diff_angle()) < 1 * 3.14 / 180

        iteration = 0

        while not condition() and iteration < self.max_iterations:

            iteration += 1

            self.strain_plane.rotate(diff_angle())
            self.analyze()

    def iterate_slope(self):
        pass

    def iterate(self):

        self.analyze()

        def condition():
            return self.force_i == self.force_e

        iteration = 0

        while not condition() and iteration < self.max_iterations:

            iteration += 1

            print(f"epsilon_o: {self.strain_plane.epsilon_o}")

            if self.steel_status.max_strain > 0.005:
                logger.debug("Se sobrepasaron las deformaciones límites del acero")
                self.report()
                return None

            if self.concrete_status.min_strain < -0.003:
                logger.debug("Se sobrepasaron las deformaciones límites del hormigón")
                self.report()
                return None

            self.iterate_depht()
            self.iterate_angle()
            self.iterate_slope()

        if iteration == self.max_iterations:
            logger.debug("Se alcanzó el límite iteraciones.")
            self.report()
            return None

        self.report()

        return self.strain_plane

    def equal_eccentricity_direction(self):

        cross_product = np.cross(self.force_e.e, self.force_i.e)

        # Verifica si los vectores de excentricidad tienen la misma dirección
        if np.allclose(cross_product, np.array([0, 0, 0])):

            # Verifica si los vectores de excentricidad tienen el mismo sentido
            dot_product = np.dot(self.force_e.e, self.force_i.e)
            if dot_product > 0:
                return True

        return False

    def is_limit_plane(self):

        concrete_limit = np.isclose(self.concrete_status.min_strain, self.concrete.limit_strain)
        steel_limit = np.isclose(self.steel_status.max_strain, self.steel.limit_strain)

        concrete_ok = self.concrete_status.min_strain > self.concrete.limit_strain
        steel_ok = self.steel_status.max_strain < self.steel.limit_strain

        return concrete_limit and steel_ok or steel_limit and concrete_ok

    def calc_Tu(self):

        # Inicia el proceso con un plano de deformación límite de tracción pura
        self.strain_plane.set_epsilon_o(self.steel.limit_strain)
        self.analyze()

        # Lanza excepción si la fuerza de tracción sobrepasa los límites de deformación del acero
        if self.force_e.N > self.force_i.N:
            raise SteelStrainExceededError(self.steel_status.max_strain, self.steel.limit_strain)

        iteration = 0

        while not (self.equal_eccentricity_direction() and self.is_limit_plane()) or iteration < self.max_iterations:
            iteration += 1

            self.iterate_depht()

    def calc_Pu(self):

        # Inicia el proceso con un plano de deformación límite de compresión pura
        self.strain_plane.set_epsilon_o(self.concrete.limit_strain)
        self.analyze()

        # Lanza excepción si la fuerza de compresión sobrepasa los límites de deformación del hormigón
        if self.force_e.N < self.force_i.N:
            raise ConcreteStrainExceededError(self.concrete_status.min_strain, self.concrete.limit_strain)

        iteration = 0

        while not (self.equal_eccentricity_direction() and self.is_limit_plane()) or iteration < self.max_iterations:
            iteration += 1

    def calc_Mu(self):
        pass

    def calc_Fu(self):

        if self.force_e.N > 0:
            self.calc_Tu()

        elif self.force_e.N < 0:
            self.calc_Pu()

        else:
            self.calc_Mu()

        self.report()

        return self.force_i

    def report(self):
        logger.info(f"External forces  : {self.force_e}")
        logger.info(f"Internal steel   : {self.steel_status.force}")
        logger.info(f"Internal concrete: {self.concrete_status.force}")
        logger.info("-----------------------------------------------------------------")
        logger.info(f"Ne - Ns - Nc     : {self.force_e - self.force_i}")
        logger.info("=================================================================")
        logger.info(f"Strain plane     : {self.strain_plane}")
        logger.info(f"max_strain: {self.steel_status.max_strain}")
        logger.info(f"min_strain: {self.concrete_status.min_strain}")
