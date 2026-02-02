import logging
import math
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt, cm
import matplotlib.colors as mcolors

from .components import Force, StrainPlane
from .fibers import RectFiber, RoundFiber, GroupFiberStatus, Fiber
from .materials import SteelMaterial, ConcreteMaterial

logger = logging.getLogger(__name__)


class ReinforcementConcreteSectionBase:

    def __init__(self, concrete, steel, bars, stirrups=None, iterations=50):

        super().__init__()

        self.concrete: ConcreteMaterial = concrete
        self.steel: SteelMaterial = steel
        self.steel_fibers: List[RoundFiber] = bars
        self.stirrups = stirrups
        self.max_iterations = iterations

        self._concrete_fibers: List[RectFiber] = []

        # Parámetros independientes de las fuerzas, los planos de deformaciones y la resolución del mallado
        self._Ag = None
        self._As = None

        self._strain_plane = StrainPlane()
        self._max_strain_steel = None
        self.force_i: Optional[Force] = None

        self._built = False

    def _build_concrete_fibers(self):
        raise NotImplementedError

    def _clean(self):
        self.strain_plane = StrainPlane()
        self._max_strain_steel = None
        self.force_i = None

    def _get_farthest_fiber_concrete(self) -> Fiber:
        """
            Obtiene la fibra más comprimida o menos traccionada.

        @return: La fibra más alejada del eje neutro.
        """
        farthest_fiber = None

        for fiber in self.get_concrete_fibers_extremes():

            fiber.distance_nn = self.strain_plane.get_dist_nn(fiber.center)
            fiber.distance_nn_cg = self.strain_plane.get_dist_nn_cg(fiber.center)

            if not farthest_fiber or fiber.distance_nn > farthest_fiber.distance_nn:
                farthest_fiber = fiber

        return farthest_fiber

    def _get_farthest_fiber_steel(self) -> Fiber:
        """
            Obtiene la fibra más traccionada o menos comprimida.

        @return: La fibra más alejada del eje neutro.
        """
        farthest_fiber = None

        for fiber in self.steel_fibers:

            fiber.distance_nn = self.strain_plane.get_dist_nn(fiber.center)
            fiber.distance_nn_cg = self.strain_plane.get_dist_nn_cg(fiber.center)

            if not farthest_fiber or fiber.distance_nn < farthest_fiber.distance_nn:
                farthest_fiber = fiber

        return farthest_fiber

    def build(self, force=False):
        if not self._built or force:
            self._clean()
            self._build_concrete_fibers()
            self._built = True
        return self

    def _calc_force_i(self):

        self.force_i = Force()

        for fiber in self.steel_fibers:
            fiber.strain = self.strain_plane.get_strain(fiber.center)
            self.force_i += fiber.force
            if self._max_strain_steel is None or fiber.strain > self._max_strain_steel:
                self._max_strain_steel = fiber.strain

        for fiber in self.concrete_fibers:
            fiber.strain = self.strain_plane.get_strain(fiber.center)
            self.force_i += fiber.force

    def _get_limits_strain(self, spp):
        """
            Obtiene las deformaciones límites del hormigón y el acero en función de una relación lineal de un parámetro
            que varía desde 0, para el plano de compresión pura, hasta 1 para el plano de tracción sin excentricidad.

        @param spp: Un valor real entre 0 y 1
        @return: Las deformaciónes específicas del hormigón y el acero correspondientes al plano límite
        """
        domain = self.steel.limit_strain - self.concrete.limit_strain
        t = 2 * domain * spp

        strain_concrete = self.concrete.limit_strain if t < domain else t - domain + self.concrete.limit_strain
        strain_steel = t + self.concrete.limit_strain if t < domain else self.steel.limit_strain

        return float(strain_concrete), float(strain_steel)

    def _get_params(self, strain_concrete, strain_steel):
        """
            Obtiene los parámetros de curvatura y desplazamiento necesarios a aplicar al plano actual para alcanzar los
            límites de deformación especificados.

        @param strain_concrete: Límite de deformación especificada para la fibra de hormigón más comprimida.
        @param strain_steel: Límite de deformación específica para la fibra de acero menos comprimida o más comprimida.
        @return: Una tupla con los valores de curvatura y deformación sobre el eje "X" en el baricentro de la sección.
        """
        ffc = self._get_farthest_fiber_concrete()  # Fibra más alejada de hormigón
        ffs = self._get_farthest_fiber_steel()  # Fibra más alejada de acero
        distance = ffc.distance_nn - ffs.distance_nn

        curvature_required = (strain_steel - strain_concrete) / distance

        strain_cg_required = curvature_required * ffc.distance_nn_cg + strain_concrete

        return curvature_required, strain_cg_required

    def increase_resolution(self, factor):
        raise NotImplementedError

    @property
    def strain_plane(self):
        return self._strain_plane

    @strain_plane.setter
    def strain_plane(self, val: StrainPlane):
        self._strain_plane.theta = val.theta
        self._strain_plane.kappa = val.kappa
        self._strain_plane.xo = val.xo

    @property
    def concrete_fibers(self):
        if not self._concrete_fibers:
            self.build()
        return self._concrete_fibers

    def get_concrete_fibers_extremes(self):
        """
            Propiedad para ser sobreescrita por las clases especializadas para devolver solo las fibras de hormigón que
            son candidatas a ser las más comprimidas y evitar así un análisis de todas las fibras.
        Returns: List[Fiber]

        """
        return self.concrete_fibers

    @property
    def Ag(self):
        """
            Área bruta del hormigón, incluyendo el área ocupada por las armaduras.
        @return: El área de la sección de hormigón, en mm²
        """
        if self._Ag is None:
            self._Ag = 0
            for fiber in self.concrete_fibers:
                self._Ag += max(fiber.area, 0)  # Para no considerar las fibras con áreas negativa
        return self._Ag

    @property
    def An(self):
        """
            Área neta del hormigón, excluyendo el área ocupada por las armaduras.
        @return: El área de la sección de hormigón, en mm²
        """
        return self.Ag - self.As

    @property
    def As(self):
        """
            Área de las armaduras.
        @return: El área de las barras de acero, en mm²
        """
        if self._As is None:
            self._As = 0
            for fiber in self.steel_fibers:
                self._As += fiber.area
        return self._As

    def phi(self):
        """
            Calcula el factor de seguridad en función del plano límite de deformación.

        @return: Un escalar.
        """

        if self._max_strain_steel >= 0.005:
            return 0.9
        elif self._max_strain_steel <= 0.002:
            return 0.65
        else:
            return np.interp(self._max_strain_steel, [0.002, 0.005], [0.65, 0.90])

    # Limit Plane

    def set_limit_plane_by_strains(self, strain_concrete, strain_steel, theta_me, max_iterations=50):
        """
            Configura un plano límite en la sección de forma tal que la fibra más comprimida del hormigón y la más
            traccionada del acero coincidan con los valores especificados y el ángulo entre el vector resultante de los
            momentos internos y el vector positivo del eje "z" coincida con el ángulo especificado o difieran en pi radianes

        @param strain_concrete: Deformación específica de la fibra de acero más traccionada o menos comprimida.
        @param strain_steel: Deformación específica de la fibra de hormigón más comprimida o menos traccionada.
        @param theta_me: Ángulo entre la resultante de los momentos y el eje "z" positivo. En radianes.
        @param max_iterations: El número máximo de iteraciones a considerar (opcional)
        """

        max_iterations = max_iterations or self.max_iterations or 200

        self._clean()

        # Construye un plano inicial para poder determinar los primeros parámetros
        self.strain_plane = StrainPlane(theta=theta_me)

        # Debido a que la condición de deformaciones límites será establecida para las fibras extremas solo se
        # verifica que el eje neutro no haya rotado luego de deformar la sección: La resultante de momentos internos
        # debe coincidir con el ángulo preestablecido.
        def condition():
            kappa, xo = self._get_params(strain_concrete, strain_steel)
            self.strain_plane.kappa = kappa
            self.strain_plane.xo = xo
            self._calc_force_i()

            # Si el plano de deformación no tiene pendiente no hay ángulo para comparar y se detiene el proceso
            if strain_steel == strain_concrete:
                return True

            # Si la fuerza es de tracción o compresión pura y no se puede definir un ángulo se asigna el del plano de
            # deformación
            theta_mi = self.force_i.theta_M or self.strain_plane.theta

            return math.isclose(0.0, (theta_me - theta_mi) % np.pi, abs_tol=1.e-2)

        iteration = 0
        while not condition():
            iteration += 1

            if iteration >= max_iterations:
                raise StopIteration(iteration)

            self.strain_plane.theta += theta_me - self.force_i.theta_M

    def set_limit_plane_by_eccentricity(self, ee, theta_me, spp_inf=0.0, spp_sup=1.0, iteration=0, max_iterations=100):
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
        @param spp_sup: El parámetro que define el plano de deformación inferior. Un número real entre ssp_inf y 1.
        @param iteration: El número iteración actual
        @param max_iterations: El número máximo de iteraciones permitidas
        """

        # Si no se especifica una excentricidad (ee=0) se genera una ambigüedad al no saber si se trata de tracción pura
        # o compresión pura.
        if ee == 0:
            raise ValueError("Se debe especificar una excentricidad distinta de cero.")

        # Se genera un punto intermedio entre los planos de deformación parametrizados para aplicar el método
        # iterativo por bisección.
        spp_mid = (spp_inf + spp_sup) / 2.0

        # Se obtienen las deformaciones específicas en función de la parametrización.
        strain_concrete, strain_steel = self._get_limits_strain(spp_mid)

        logger.debug("%d (%f): [%f ; %f]", iteration, spp_mid, strain_concrete, strain_steel)

        # Se aplica el plano por deformaciones y se obtiene la excentricidad de las fuerzas internas.
        self.set_limit_plane_by_strains(strain_concrete, strain_steel, theta_me)

        # Si la excentricidad es infinita estamos en un caso de flexión pura
        ei = self.force_i.e
        if ee == np.inf or ee == -np.inf:
            threshold = min(self.get_Pnt(), -self.get_Pnc()) * 0.01
            condition = np.isclose(float(self.force_i.N), 0.0, atol=threshold)
        else:
            condition = np.isclose(ee, ei, rtol=0.01)

        # Si coinciden las excentricidades se termina la iteración
        if condition:
            return

        iteration += 1

        # Si se alcanzó el límite de iteraciones se cancela el proceso
        if iteration > max_iterations:
            logger.warning(
                "Se alcanzó el número de iteraciones máximas con el plano %f ; %f",
                strain_concrete, strain_steel
            )
            raise StopIteration

        # Si no se encontró el equilibrio al acercarse los límites superior e inferior se aumenta la resolución
        if np.isclose(spp_inf, spp_sup):
            spp_inf = spp_inf * 0.2
            spp_sup = 0.8 * spp_sup + 0.2
            self.increase_resolution(2)
            logger.debug("Duplicación de la resolución")

        # Se realiza una comparación doble debido a la discontinuidad de la función para determinar si el parámetro
        # medio se utilizará como límite superior o inferior en la siguiente iteración
        elif ee < 0 and (ei < ee or ei > 0) or ee > 0 and (ee >= ei >= 0):
            spp_sup = spp_mid

        else:
            spp_inf = spp_mid

        self.set_limit_plane_by_eccentricity(
            ee, theta_me, spp_inf=spp_inf, spp_sup=spp_sup, iteration=iteration, max_iterations=max_iterations
        )

    # Forces

    def get_Pd_max(self):
        """
            Obtiene la resistencia de diseño a compresión límite para la sección.

        @return: Un escalar representando la resistencia límite, en N.
        """

        coef = 0.85 if self.stirrups and self.stirrups.stirrup_type == 2 else 0.80

        return -coef * self.phi() * (0.85 * self.concrete.fpc * (self.Ag - self.As) + self.steel.fy * self.As)

    def get_Pnc(self):
        """
            Obtiene la resistencia nominal a compresión de la sección.
        @return: Un escalar representando la resistencia de la sección, en N.
        """
        return -0.85 * self.concrete.fpc * (self.Ag - self.As) - self.steel.fy * self.As

    def get_Pnt(self):
        """
            Obtiene la resistencia nominal a tracción de la sección.
        @return: Un escalar representando la resistencia de la sección, en N.
        """
        return self.steel.fy * self.As

    def get_nominal_force(self, force: Force):

        ee = force.e
        theta = force.theta_M or 0.0

        if ee != 0:
            self.set_limit_plane_by_eccentricity(ee, theta)
        elif force.N > 0:
            self.set_limit_plane_by_strains(self.steel.limit_strain, self.steel.limit_strain, 0.0)
        else:
            self.set_limit_plane_by_strains(self.concrete.limit_strain, self.concrete.limit_strain, 0.0)

        return self.force_i

    def get_design_force(self, force: Force):

        # Aquí el orden de los factores sí importa: es necesario calcular la fuerza nominal primero para poder hacer
        # uso de la función phi
        design = self.get_nominal_force(force) * self.phi()

        # Se limita la resistencia a compresión del hormigón
        limit_factor = min(1, self.get_Pd_max() / design.N) if design.N < 0 else 1

        return limit_factor * design

    def get_forces(self, theta_me=0, number=32) -> List[Force]:
        """
            Obtiene una lista de fuerzas sobre las curvas de interacción que representan la resistencia
            nominal para distintas excentricidades. Se debe especificar explícitamente el
            ángulo que determina el meridiano sobre el que se construirá el diagrama.

        @param theta_me: Un ángulo positivo medido en sentido antihorario desde el eje positivo Mz que representa la
        inclinación del momento resultante [radianes]
        @param number: Cantidad de puntos a representar
        @return: Una lista con las fuerzas nominales.
        """
        forces = []
        self.build()
        for value in range(number+1):
            spp = value / number
            self.set_limit_plane_by_strains(*self._get_limits_strain(spp), theta_me)
            forces.append(self.force_i)

        return forces

    def get_rel(self, force):
        return force.N / self.get_design_force(force).N
