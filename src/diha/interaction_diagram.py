import logging
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from .components import Force, StrainPlane, ForceExt
from .fibers import RectFiber, RoundFiber, GroupFiberStatus, Fiber
from .materials import SteelMaterial, ConcreteMaterial
from .utils import calc_angle_yz

logger = logging.getLogger(__name__)

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

    def __init__(self, concrete, steel, bars, stirrups=None, iterations=10):

        super().__init__()

        self.max_iterations= iterations

        self.strain_plane = StrainPlane()

        self.concrete: ConcreteMaterial = concrete
        self.steel: SteelMaterial = steel

        self.steel_fibers: List[RoundFiber] = bars
        self._concrete_fibers: List[RectFiber] = []
        self.stirrups = stirrups

        self.force_e = None
        self.theta_me = None
        self.force_i = Force()
        self.theta_mi = None

        self.concrete_status = GroupFiberStatus()
        self.steel_status = GroupFiberStatus()

        # Ángulo desde el plano zx hacia el semiplano que contiene al eje x y sobre el cual se define la curva de
        # interacción.
        self._Ag = None
        self._As = None

        self._built = False

    def _build_concrete_fibers(self):
        raise NotImplementedError

    def build(self, force=False):
        if not self._built or force:
            self._build_concrete_fibers()
            self._built = True

    @property
    def concrete_fibers(self):
        if not self._concrete_fibers:
            self.build()
        return self._concrete_fibers

    @property
    def Ag(self):
        if self._Ag is None:
            self._Ag = 0
            for fiber in self.concrete_fibers:
                self._Ag += max(fiber.area, 0)
        return self._Ag

    @property
    def An(self):
        return self.Ag - self.As

    @property
    def As(self):
        if self._As is None:
            self._As = 0
            for fiber in self.steel_fibers:
                self._As += fiber.area
        return self._As

    @staticmethod
    def phi(strain_steel):
        if strain_steel >= 0.005:
            return 0.9
        elif strain_steel <= 0.002:
            return 0.65
        else:
            return np.interp(strain_steel, [0.002, 0.005], [0.65, 0.90])

    def Pn_max(self, strain_steel):

        phi = self.phi(strain_steel)

        coef = 0.85 if self.stirrups and self.stirrups.type == 2 else 0.80

        return -coef * phi * (0.85 * self.concrete.fpc * (self.Ag - self.As) + self.steel.fy * self.As) * 1e-3

    def analyze(self):

        self.build()

        for fiber in self.steel_fibers:
            fiber.strain = self.strain_plane.get_strain(fiber.point)

        for fiber in self.concrete_fibers:
            fiber.strain = self.strain_plane.get_strain(fiber.point)

        self.concrete_status.update(self.concrete_fibers)
        self.steel_status.update(self.steel_fibers)

        self.force_i = self.concrete_status.force + self.steel_status.force

    def get_farthest_fiber_compression(self, fibers: List[Fiber], strain_plane=None):

        strain_plane = strain_plane or self.strain_plane

        farthest_fiber_compression = None

        for fiber in fibers:

            fiber.distance_nn = strain_plane.get_dist_calc(fiber.point)
            fiber.distance_nn_cg = strain_plane.get_dist_nn_cg(fiber.point)

            if not farthest_fiber_compression or fiber.distance_nn < farthest_fiber_compression.distance_nn:
                farthest_fiber_compression = fiber

        return farthest_fiber_compression

    def get_farthest_fiber_tension(self, fibers: List[Fiber], strain_plane=None):

        strain_plane = strain_plane or self.strain_plane

        farthest_fiber_tension = None

        for fiber in fibers:

            fiber.distance_nn = strain_plane.get_dist_calc(fiber.point)
            fiber.distance_nn_cg = strain_plane.get_dist_nn_cg(fiber.point)

            if not farthest_fiber_tension or fiber.distance_nn > farthest_fiber_tension.distance_nn:
                farthest_fiber_tension = fiber

        return farthest_fiber_tension

    def get_limits_strain(self, param):
        """
            Obtiene las deformaciones límites del hormigón y el acero en función de una relación lineal de un parámetro
            que varía desde 0, para el plano de compresión pura, hasta 1 para el plano de tracción sin excentricidad.

        @param param: Un valor real entre 0 y 1
        @return: Las deformaciónes específicas del hormigón y el acero correspondientes al plano límite
        """
        domain = self.steel.limit_strain - self.concrete.limit_strain
        t = 2 * domain * param

        strain_concrete = self.concrete.limit_strain if t < domain else t - domain + self.concrete.limit_strain
        strain_steel = t + self.concrete.limit_strain if t < domain else self.steel.limit_strain

        return strain_concrete, strain_steel

    def _get_params(self, strain_concrete, strain_steel):
        ffc = self.get_farthest_fiber_compression(self.concrete_fibers)  # Fibra más alejada de hormigón
        ffs = self.get_farthest_fiber_tension(self.steel_fibers)  # Fibra más alejada de acero
        distance = ffs.distance_nn - ffc.distance_nn

        curvature_required = (strain_steel - strain_concrete) / distance

        strain_cg_required = curvature_required * -ffc.distance_nn_cg + strain_concrete

        return curvature_required, strain_cg_required

    def set_limit_plane_by_strains(self, strain_concrete, strain_steel, theta_me=None):
        """
            Configura un plano límite en la sección de forma tal que la fibra más comprimida del hormigón y la más
            traccionada del acero coincidan con los valores especificados y el ángulo entre el vector resultante de los
            momentos internos y el vector positivo del eje z coincida con el ángulo especificado.

        @param theta_me: Ángulo entre la resultante de los momentos y el eje z positivo. En radianes.
        """

        # Configura el ángulo del plano sobre el que se construirá el diagrama de interacción
        theta_me = theta_me or self._get_theta_me(self.force_e)

        # Construye un plano inicial para poder determinar los primeros parámetros
        self.strain_plane = StrainPlane(theta=theta_me)

        # Debido a que la condición de deformaciones límites ya ha sido establecida para los elementos extremos solo se
        # verifica que el eje neutro no haya rotado al deformar la sección: La resultante de momentos internos debe
        # coincidir con el ángulo definido.
        def condition():
            kappa, xo = self._get_params(strain_concrete, strain_steel)
            self.strain_plane.kappa = kappa
            self.strain_plane.xo = xo
            self.analyze()

            theta_mi = self.force_i.theta_M or theta_me

            if theta_mi is None:
                return True

            return np.isclose(theta_me, theta_mi, rtol=1.e-5, atol=1.e-8)

        iteration = 0
        while not condition() and iteration < self.max_iterations:
            iteration += 1
            delta_theta = theta_me - self.force_i.theta_M
            self.strain_plane.theta += delta_theta

        if iteration >= self.max_iterations:
            raise StopIteration

    @staticmethod
    def _get_theta_me(force=None):
        if force and not np.isclose(0, np.linalg.norm(force.M), atol=1e-6):
            return calc_angle_yz(np.array([0, 0, 1]), force.M)
        return 0

    @staticmethod
    def _get_e(force):
        if force:
            if force.N != 0:
                return np.linalg.norm(force.M) / force.N
            return np.inf
        return 0

    def set_limit_plane_by_eccentricity(self, ee, theta_me=None, **kwargs):
        theta_me = theta_me or self._get_theta_me(self.force_e)

        # Si no se especifica una excentricidad se genera una ambiguedad al no saber si se trata de tracción pura o
        # compresión pura.
        if ee == 0:
            raise ValueError("No se pudo resolver un valor de excentricidad distinto de cero")

        spp_inf = kwargs.get("spp_inf", 0.0)
        spp_sup = kwargs.get("spp_sup", 1.0)
        kwargs["iteration"] = kwargs.get("iteration", 0) + 1

        # Se genera un punto intermedio entre los planos de deformación parametrizados para aplicar el método
        # iterativo por bisección.
        spp_mid = (spp_inf + spp_sup) / 2

        # Se obtienen las deformaciones específicas en función de la parametrización.
        strain_concrete, strain_steel = self.get_limits_strain(spp_mid)

        # Se aplica el plano y obtiene la excentricidad de las fuerzas internas.
        self.set_limit_plane_by_strains(strain_concrete, strain_steel, theta_me=theta_me)
        ei = self._get_e(self.force_i)
        logger.debug("%d (%f): [%f ; %f]", kwargs["iteration"], spp_mid, strain_concrete, strain_steel)

        # Si las excentricidades son suficientemente iguales se considera la fuerza interna como la resistencia nominal
        # para la excentricidad de la carga
        if np.isclose(ee, ei, rtol=0.01):
            return ForceExt(self.force_i, strain_steel)

        # Si no son iguales las excentricidades y se alcanzó el límite de interacciones se cancela el procedimiento
        if kwargs["iteration"] > kwargs.get('max_iterations', 200):
            logger.warning(
                "Se alcanzó el número de iteraciones máximas con el plano %f ; %f",
                strain_concrete, strain_steel
            )
            raise StopIteration

        if ee < 0 and (ei < ee or ei > 0) or ee >= 0 and (ee >= ei >= 0):
            kwargs['spp_sup'] = spp_mid
        else:
            kwargs['spp_inf'] = spp_mid

        return self.set_limit_plane_by_eccentricity(ee, theta_me=theta_me, **kwargs)

    def get_forces(self, theta_me=None, number=20) -> List[ForceExt]:
        """
            Obtiene una lista de fuerzas sobre las curvas de interacción que representan la resistencia
            nominal sobre una línea de fuerzas con la misma excentricidad. Se puede especificar explícitamente el
            ángulo que determina el meridiano sobre el que se construirá el diagrama o se puede indicar en forma
            implícita a través de una fuerza externa.

        @param theta_me: Un ángulo positivo medido en sentido antihorario desde el eje positivo Mz que representa la
        inclinación del momento resultante [radianes]
        @param number: Cantidad de puntos a representar
        @return: Una lista con las fuerzas nominales.
        """
        forces = []
        self.build()
        for ee in [-0.1, -0.2, -0.3]:
            try:
                force = self.set_limit_plane_by_eccentricity(theta_me=theta_me, ee=ee)
                forces.append(force)
            except StopIteration:
                logger.warning("Se alcanzó el limite de iteraciones")

        return forces

    def plot_diagram_2d(self, theta_me=0):
        nominal = []
        design = []

        strain_concrete = -0.003
        strain_steel = -0.003
        delta_strain = 0.0002

        while strain_concrete <= 0.005:
            self.set_limit_plane_by_strains(strain_concrete, strain_steel, theta_me=theta_me)

            M = np.linalg.norm(self.force_i.M) * 1e-6
            N = self.force_i.N * 1e-3

            if np.isclose(strain_steel, 0.002, atol=delta_strain / 2):
                plt.plot([0, M], [0, N], linestyle='--', color='gray')

            if (np.isclose(strain_steel, 0.005, atol=delta_strain / 2) and
                    np.isclose(strain_concrete, -0.003, atol=delta_strain / 2)):
                plt.plot([0, M], [0, N], linestyle='--', color='gray')

            nominal.append([M, N])

            factor = self.phi(strain_steel)
            design.append([factor * M, max(self.Pn_max(factor), factor * N)])

            if strain_steel >= 0.005:
                strain_concrete += delta_strain
            else:
                strain_steel += delta_strain

        x, y = zip(*nominal)
        plt.plot(x, y, marker='', linestyle='-', color='g', label='Nn-Mn')

        x, y = zip(*design)
        plt.plot(x, y, marker='', linestyle='-', color='r', label='Nd-Md')

        plt.xlabel('M [kNm]')
        plt.ylabel('N [kN]')

        plt.gca().invert_yaxis()

        plt.title(f'Diagrama de interacción - \u03B8={np.degrees(theta_me)}')
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