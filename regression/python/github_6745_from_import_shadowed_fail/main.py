from shapes import Shape


class Shape:

    def __init__(self) -> None:
        self.edges: int = 3

    def total(self) -> int:
        return self.edges


class Square(Shape):

    def get(self) -> int:
        return self.total()


s = Square()
assert s.get() == 4
