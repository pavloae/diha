import logging
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from .components import Force, StrainPlane
from .fibers import RectFiber, RoundFiber, GroupFiberStatus, Fiber
from .materials import SteelMaterial, ConcreteMaterial

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

        # Ángulo desde el plano zx hacia el semiplano que contiene al eje "x" y sobre el cual se define la curva de
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

    @staticmethod
    def get_farthest_fiber(fibers: List[Fiber], strain_plane: StrainPlane, compression=True)->Fiber:
        """
            Dado un conjunto de fibras y un plano de deformaciones obtiene la fibra a compresión o tracción más alejada
            del eje neutro.
        @param fibers: Una lista de fibras
        @param strain_plane: Un plano de deformaciones
        @param compression: Especifica si se obtendrá la fibra más comprimida (o menos traccionada): compression=True;
                            o la fibra más traccionada (o menos comprimida) compression=False.
        @return: La fibra más alejada del eje neutro.
        """
        farthest_fiber = None

        for fiber in fibers:

            fiber.distance_nn = strain_plane.get_dist_calc(fiber.point)
            fiber.distance_nn_cg = strain_plane.get_dist_nn_cg(fiber.point)

            if not farthest_fiber or (compression == fiber.distance_nn <= farthest_fiber.distance_nn):
                farthest_fiber = fiber

        return farthest_fiber

    @staticmethod
    def _get_e(force):
        if force:
            if force.N != 0:
                return np.linalg.norm(force.M) / force.N
            return np.inf
        return 0

    def Pn_max(self, strain_steel):

        phi = self.phi(strain_steel)

        coef = 0.85 if self.stirrups and self.stirrups.stirrup_type == 2 else 0.80

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
        ffc = self.get_farthest_fiber(self.concrete_fibers, self.strain_plane, compression=True)  # Fibra más alejada de hormigón
        ffs = self.get_farthest_fiber(self.steel_fibers, self.strain_plane, compression=False)  # Fibra más alejada de acero
        distance = ffc.distance_nn - ffs.distance_nn

        curvature_required = (strain_steel - strain_concrete) / distance

        strain_cg_required = curvature_required * ffc.distance_nn_cg + strain_concrete

        return curvature_required, strain_cg_required

    def set_limit_plane_by_strains(self, strain_concrete, strain_steel, theta_me=0, max_iterations=None):
        """
            Configura un plano límite en la sección de forma tal que la fibra más comprimida del hormigón y la más
            traccionada del acero coincidan con los valores especificados y el ángulo entre el vector resultante de los
            momentos internos y el vector positivo del eje "z" coincida con el ángulo especificado.

        @param strain_steel: Deformación específica de la fibra de hormigón más comprimida o menos traccionada.
        @param strain_concrete: Deformación específica de la fibra de acero más traccionada o menos comprimida.
        @param theta_me: Ángulo entre la resultante de los momentos y el eje "z" positivo. En radianes.
        @param max_iterations:
        """

        max_iterations = max_iterations or self.max_iterations or 200

        # Construye un plano inicial para poder determinar los primeros parámetros
        self.strain_plane = StrainPlane(theta=theta_me)

        # Debido a que la condición de deformaciones límites será establecida para las fibras extremas solo se
        # verifica que el eje neutro no haya rotado luego de deformar la sección: La resultante de momentos internos
        # debe coincidir con el ángulo preestablecido.
        def condition():
            kappa, xo = self._get_params(strain_concrete, strain_steel)
            self.strain_plane.kappa = kappa
            self.strain_plane.xo = xo
            self.analyze()

            # Si la fuerza es de tracción o compresión pura y no se puede definir un ángulo se asigna el del plano de
            # deformación
            theta_mi = self.force_i.theta_M or self.strain_plane.theta

            return np.isclose(theta_me, theta_mi, rtol=1.e-5, atol=1.e-8)

        iteration = 0
        while not condition():
            iteration += 1

            if iteration >= max_iterations:
                raise StopIteration

            delta_theta = theta_me - self.force_i.theta_M
            self.strain_plane.theta += delta_theta

    def set_limit_plane_by_eccentricity(self, ee, theta_me, spp_inf=0.0, spp_sup=1.0, iteration=0, max_iterations=200):
        """
            Configura un plano límite para la sección en donde la resultante de las fuerzas internas tenga una
            excentricidad igual a la indicada y el vector de momentos forme un ángulo con el eje "z" igual al indicado.

            Es decir, determina el plano que define el punto donde se intersectan la recta de ee = M / N con la curva
            del diagrama de interacción para una flexión oblicua determinada por theta_me.

            La parametrización de los distintos planos límites que van desde spp=0 para la sección sometida a compresión
            pura, hasta spp=1 para la sección sometida a tracción pura, determina una excentricidad de las fuerzas
            internas que varía en función de dicho parámetro. Siendo:

            ee(spp=0) = 0           (Para el caso de compresión pura)

            ee(spp<spp_lim) < 0     (Tramo continuo de la función)

            lim ee(spp) = -∞        (Para el caso de flexión pura desde la compresión)
              spp->+spp_lim

                ------ Discontinuidad de la función en spp_lim cuando N=0 ----------

            lim ee(spp) = +∞        (Para el caso de flexión pura desde la tracción)
              spp->+spp_lim

            ee(spp>spp_lim) > 0     (Tramo continuo de la función)

            ee(spp=1) = 0           (Para el caso de tracción pura)

            El cálculo del plano se realiza mediante el método de la bisección, utilizando como extremos iniciales los
            planos de compresión y tracción pura.

        @param ee: Excentricidad de las fuerzas externas ee = M / N. Adoptándose M siempre positiva, el signo de la
                    excentricidad será positivo con una fuerza de tracción y negativo con una fuerza de compresión.
        @param theta_me: Ángulo que forma el vector de momentos exteriores con el eje "z" positivo. [rad]
        @param spp_inf: El parámetro que define el plano de deformación inferior. Un número real entre 0 y 1.
        @param spp_sup: El parámetro que define el plano de deformación inferior. Un número real entre 0 y 1.
        @param iteration: El número iteración actual
        @param max_iterations: El número máximo de iteraciones permitidas
        """

        # Si no se especifica una excentricidad (ee=0) se genera una ambigüedad al no saber si se trata de tracción pura
        # o compresión pura.
        if ee == 0:
            raise ValueError("Se debe especificar una excentricidad distinta de cero.")

        # Se genera un punto intermedio entre los planos de deformación parametrizados para aplicar el método
        # iterativo por bisección.
        spp_mid = (spp_inf + spp_sup) / 2

        # Se obtienen las deformaciones específicas en función de la parametrización.
        strain_concrete, strain_steel = self.get_limits_strain(spp_mid)

        # Se aplica el plano por deformaciones y se obtiene la excentricidad de las fuerzas internas.
        self.set_limit_plane_by_strains(strain_concrete, strain_steel, theta_me)
        ei = self.force_i.e

        iteration += 1
        logger.debug("%d (%f): [%f ; %f]", iteration, spp_mid, strain_concrete, strain_steel)

        # Si las excentricidades no son suficientemente iguales se continúa la iteración
        if not np.isclose(ee, ei, rtol=0.01):

            # Si se alcanzó el límite de iteraciones se cancela el proceso
            if iteration > max_iterations:
                logger.warning(
                    "Se alcanzó el número de iteraciones máximas con el plano %f ; %f",
                    strain_concrete, strain_steel
                )
                raise StopIteration

            # Se realiza una comparación doble debido a la discontinuidad de la función para determinar si el parámetro
            # medio se utilizará como límite superior o inferior en la siguiente iteración
            if ee < 0 and (ei < ee or ei > 0) or ee >= 0 and (ee >= ei >= 0):
                spp_sup = spp_mid
            else:
                spp_inf = spp_mid

            self.set_limit_plane_by_eccentricity(
                ee, theta_me, spp_inf=spp_inf, spp_sup=spp_sup, iteration=iteration, max_iterations=max_iterations
            )

    def get_forces(self, theta_me=0, number=20) -> List[Force]:
        """
            Obtiene una lista de fuerzas sobre las curvas de interacción que representan la resistencia
            nominal para distintas excentricidades. Se puede especificar explícitamente el
            ángulo que determina el meridiano sobre el que se construirá el diagrama o se puede indicar en forma
            implícita a través de una fuerza externa sobre la sección.

        @param theta_me: Un ángulo positivo medido en sentido antihorario desde el eje positivo Mz que representa la
        inclinación del momento resultante [radianes]
        @param number: Cantidad de puntos a representar
        @return: Una lista con las fuerzas nominales.
        """
        forces = []
        self.build()
        for value in range(number+1):
            self.set_limit_plane_by_strains(*self.get_limits_strain(value / number), theta_me)
            forces.append(self.force_i)

        return forces

    def plot_diagram_2d(self, theta_me=0, points=42):
        nominal = []
        design = []

        for val in range(points + 1):
            strain_concrete, strain_steel = self.get_limits_strain(val / points)

            self.set_limit_plane_by_strains(strain_concrete, strain_steel, theta_me)

            M, N = np.linalg.norm(self.force_i.M) * 1e-6, self.force_i.N * 1e-3

            nominal.append([M, N])

            factor = self.phi(strain_steel)
            design.append([factor * M, max(self.Pn_max(factor), factor * N)])

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