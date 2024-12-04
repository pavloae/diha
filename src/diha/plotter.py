from matplotlib import pyplot as plt, patches as patches


class PlotterMixin:

    def _get_concrete_fibers(self):
        raise self.concrete_fibers

    def _get_steel_fibers(self):
        return self.steel_fibers

    def _plot_contour(self, ax):
        raise NotImplementedError

    def _set_limits(self, ax):
        pass

    def plot_section(self):

        fig, ax = plt.subplots(figsize=(6, 8))

        # Dibujar elementos de hormigón (rectángulos)
        for fiber in self._get_concrete_fibers():
            y0, z0 = fiber.get_bottom_left()
            rect = patches.Rectangle(
                (z0, y0), fiber.dy, fiber.dz, edgecolor="gray", facecolor="lightblue", alpha=0.5
            )
            ax.add_patch(rect)

        # Dibujar contorno de la sección
        self._plot_contour(ax)

        # Dibujar armaduras como círculos
        for fiber in self._get_steel_fibers():
            y, z = fiber.center
            circle = plt.Circle((z, y), fiber.diam / 2, color='red', alpha=0.7)
            ax.add_patch(circle)

        # Configurar gráfico
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Z (mm)")
        ax.set_ylabel("Y (mm)")
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.title("Sección de Hormigón Armado")
        plt.grid(False)
        plt.autoscale()
        plt.show()
