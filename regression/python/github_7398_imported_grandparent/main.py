import shapes
from shapes import Shape


class Square(Shape):

    def get(self) -> int:
        return self.total()


class Grid(shapes.Panel):

    def get(self) -> int:
        return self.total()


s = Square()
g = Grid()
assert s.get() == 4
assert g.get() == 4
