# The local Shape and shapes.Shape are distinct classes (#7397).
import shapes


class Shape:
    def __init__(self) -> None:
        self.edges: int = 3

    def total(self) -> int:
        return self.edges


a = Shape()
b = shapes.Shape()
assert a.total() == 3
assert b.total() == 4
assert b.twice(shapes.make()) == 8
