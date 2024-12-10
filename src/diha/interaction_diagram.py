from typing import List

import numpy as np
from matplotlib import pyplot as plt

from .components import Force, StrainPlane, ForceExt
from .fibers import RectFiber, RoundFiber, GroupFiberStatus, Fiber
from .materials import SteelMaterial, ConcreteMaterial
from .utils import calc_angle_yz


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


class ReinforcementConcreteSection:

    def __init__(self, concrete, steel, bars, stirrups=None, N=0, My=0, Mz=0, max_initial_strain=0.001, iterations=10):

        super().__init__()

        self.max_iterations= iterations

        self.strain_plane = StrainPlane()

        self.concrete: ConcreteMaterial = concrete
        self.steel: SteelMaterial = steel

        self.steel_fibers: List[RoundFiber] = bars
        self._concrete_fibers: List[RectFiber] = []

        self.stirrups = stirrups

        self.max_initial_strain = max_initial_strain

        self.force_e = Force(N, My, Mz)
        self.theta_me = None

        self.force_i = Force()
        self.theta_mi = None

        self.concrete_status = GroupFiberStatus()
        self.steel_status = GroupFiberStatus()

        # Ángulo desde el plano zx hacia el semiplano que contiene al eje x y sobre el cual se define la curva de
        # interacción.
        self.theta = 0

        self._built = False

    def build(self, force=False):
        if not self._built or force:
            self._build_concrete_fibers()
            self._built = True

    @property
    def concrete_fibers(self):
        if not self._concrete_fibers:
            self.build()
        return self._concrete_fibers

    def factor(self, strain_steel):
        if strain_steel >= 0.005:
            return 0.9
        elif strain_steel <= 0.002:
            return 0.65
        else:
            return np.interp(strain_steel, [0.002, 0.005], [0.65, 0.90])

    def get_Pn_max(self, fi):
        Ag = 0
        As = 0

        for fiber in self.concrete_fibers:
            Ag += fiber.area

        for fiber in self.steel_fibers:
            As += fiber.area

        coef = 0.8 if not self.stirrups or self.stirrups.type == 1 else 0.85

        return -coef * fi * (0.85 * self.concrete.fpc * (Ag - As) + self.steel.fy * As) * 1e-3

    def _build_concrete_fibers(self):
        raise NotImplementedError

    def analyze(self):

        self.build()

        for fiber in self.steel_fibers:
            fiber.strain = self.strain_plane.get_strain(fiber.point)

        for fiber in self.concrete_fibers:
            fiber.strain = self.strain_plane.get_strain(fiber.point)

        self.concrete_status.update(self.concrete_fibers)
        self.steel_status.update(self.steel_fibers)

        self.force_i = self.concrete_status.force + self.steel_status.force

    def get_farthest_fiber_compression(self, fibers: List[Fiber]):

        farthest_fiber_compression = None

        for fiber in fibers:

            fiber.distance_nn = self.strain_plane.get_dist_calc(fiber.point)
            fiber.distance_nn_cg = self.strain_plane.get_dist_nn_cg(fiber.point)

            if not farthest_fiber_compression or fiber.distance_nn < farthest_fiber_compression.distance_nn:
                farthest_fiber_compression = fiber

        return farthest_fiber_compression

    def get_farthest_fiber_tension(self, fibers: List[Fiber]):

        farthest_fiber_tension = None

        for fiber in fibers:

            fiber.distance_nn = self.strain_plane.get_dist_calc(fiber.point)
            fiber.distance_nn_cg = self.strain_plane.get_dist_nn_cg(fiber.point)

            if not farthest_fiber_tension or fiber.distance_nn > farthest_fiber_tension.distance_nn:
                farthest_fiber_tension = fiber

        return farthest_fiber_tension

    def try_limit_plane(self, strain_concrete, strain_steel, theta_me=None):
        """
            Configura un plano límite en la sección de forma tal que la fibra más comprimida del hormigón y la más
            traccionada del acero coincidan con los valores especificados y el ángulo entre el vector resultante de los
            momentos internos y el vector positivo del eje z coincida con el ángulo especificado.

        @param strain_concrete: Deformación específica para la fibra de hormigón más comprimida o menos traccionada.
        @param strain_steel: Deformación específica para la fibra de acer más traccionada o menos comprimida.
        @param theta_me: Ángulo entre la resultante de los momentos y el eje z positivo. En radianes.
        """

        if not theta_me and self.force_e and np.linalg.norm(self.force_e.M) != 0:
            theta_me = calc_angle_yz(np.array([0, 0, 1]), self.force_e.M)

        theta_me = theta_me or 0

        self.theta_me = theta_me
        self.strain_plane = StrainPlane(theta=theta_me)

        def get_params():
            ffc = self.get_farthest_fiber_compression(self.concrete_fibers) # Fibra más alejada de hormigón
            ffs = self.get_farthest_fiber_tension(self.steel_fibers) # Fibra más alejada de acero

            delta_strain = strain_steel - strain_concrete
            distance = ffs.distance_nn - ffc.distance_nn

            curvature_required = delta_strain / distance

            strain_required = curvature_required * -ffc.distance_nn_cg + strain_concrete

            return curvature_required, strain_required

        def condition():
            kappa, xo = get_params()
            self.strain_plane = StrainPlane(theta=self.theta_me, kappa=kappa, xo=xo)
            self.analyze()
            self.theta_mi = calc_angle_yz(np.array([0, 0, 1]), self.force_i.M)
            return np.isclose(self.theta_mi, theta_me)

        iteration = 0
        while not condition() and iteration < self.max_iterations:
            iteration += 1

            delta_theta = theta_me - self.theta_mi
            self.strain_plane.theta += delta_theta

            kappa, xo = get_params()
            self.strain_plane = StrainPlane(theta=theta_me, kappa=kappa, xo=xo)
            self.analyze()

        if iteration >= self.max_iterations:
            raise StopIteration

    def get_forces(self, moment_angle=None, delta_strain=0.0002, strain_concrete=-0.003, strain_steel=-0.003) -> List[ForceExt]:
        """
            Obtiene una lista de fuerzas extendidas sobre las curvas de interacción que representan la resistencia
            nominal sobre una línea de fuerzas con la misma excentricidad. Se puede especificar explícitamente el
            ángulo que determina el meridiano sobre el que se construirá el diagrama o se puede indicar en forma
            implícita a través de una fuerza externa.

        @param moment_angle: Un ángulo positivo medido en sentido antihorario desde el eje positivo Mz que representa la
        inclinación del momento resultante [radianes]
        @param delta_strain: El incremento de deformación unitaria a utilizar en cada paso.
        @param strain_concrete: Deformación inicial de la fibra más comprimida de hormigón
        @param strain_steel: Deformación inicial de la fibra más tracionada de acero
        @return:
        """

        forces = []

        self.build()

        count = 0
        while strain_steel <= 0.005 and strain_concrete < 0.005:

            try:
                self.try_limit_plane(strain_concrete, strain_steel, moment_angle)
                forces.append(ForceExt(self.force_i, strain_steel, strain_concrete, self.factor(strain_steel)))
            except StopIteration:
                count += 1

            if strain_steel < 0.005:
                strain_steel += delta_strain
            else:
                strain_steel = 0.005
                strain_concrete += delta_strain

        print(f"Planos construidos: {len(forces)} - Planos fallidos: {count}")
        return forces

    def plot_diagram(self, theta=0):
        nominal = []
        design = []

        delta_strain = 0.0002
        for force_ext in self.get_forces(delta_strain=delta_strain):

            M = np.linalg.norm(force_ext.M) * 1e-6
            N = force_ext.N * 1e-3

            ss = force_ext.strain_steel

            # if np.isclose(ss, 0.002, atol=delta_strain/2):
            #     plt.plot([0, M], [0, N], linestyle='--', color='gray')
            #
            # if (np.isclose(ss, 0.005, atol=delta_strain/2) and
            #         np.isclose(force_ext.strain_concrete, -0.003, atol=delta_strain/2)):
            #     plt.plot([0, M], [0, N], linestyle='--', color='gray')

            plt.plot([0, M], [0, N], linestyle='--', color='gray')

            nominal.append([M, N])

            factor = self.factor(ss)
            design.append([factor * M, max(self.get_Pn_max(factor), factor * N)])

        x, y = zip(*nominal)
        plt.plot(x, y, marker='', linestyle='-', color='g', label='Nn-Mn')

        x, y = zip(*design)
        plt.plot(x, y, marker='', linestyle='-', color='r', label='Nd-Md')

        plt.xlabel('M [kNm]')
        plt.ylabel('N [kN]')

        plt.gca().invert_yaxis()

        plt.title(f'Diagrama de interacción - \u03B8={np.degrees(theta)}')
        plt.legend()
        plt.grid(True)
        plt.autoscale()
        plt.show()

    def plot_section(self):

        fig, ax = plt.subplots(figsize=(6, 8))

        self.build()

        # Dibuja elementos de hormigón
        for fiber in self.concrete_fibers:
            fiber.plot(ax)

        # Dibuja armaduras
        for fiber in self.steel_fibers:
            fiber.plot(ax)

        # Configura gráfico
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Z (mm)")
        ax.set_ylabel("Y (mm)")
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

        plt.gca().invert_xaxis()
        plt.title(f"{self.__class__.__name__}")
        plt.grid(False)
        plt.autoscale()
        plt.show()