class Point2D:
    def __init__(self, y, z):
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


class Point3D(Point2D):

    def __init__(self, x, y, z):
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