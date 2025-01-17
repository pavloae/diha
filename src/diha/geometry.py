class Point2D:
    def __init__(self, y, z):
        """
            Clase para representar un punto en un plano en dos dimensiones.

        @param y: Coordenada sobre el eje vertical y, en mm.
        @param z: Coordenadas sobre el eje horizontal z, en mm.
        """
        self.y = y
        self.z = z

    def __getitem__(self, index):
        if index == 0:
            return self.y
        elif index == 1:
            return self.z
        else:
            raise IndexError("Point only supports indices 0 (y) and 1 (z)")

    def __repr__(self):
        return f"Point(y={self.y}, z={self.z})"

    def __str__(self):
        return f"[{self.y}, {self.z}]"


class Point3D(Point2D):

    def __init__(self, x, y, z):
        """
            Clase para representar un punto en un plano en tres dimensiones.

        @param x: Coordenada sobre el eje perpendicular al plano x, en mm.
        @param y: Coordenadas sobre el eje horizontal y, en mm.
        @param z: Coordenadas sobre el eje horizontal z, en mm.
        """
        super().__init__(y, z)
        self.x = x

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Point only supports indices 0 (x), 1 (y) and 2 (z)")

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

    def __str__(self):
        return f"[{self.x}, {self.y}, {self.z}]"

