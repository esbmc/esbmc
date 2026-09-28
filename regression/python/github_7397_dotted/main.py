# pkg.shapes.Shape is renamed through the dotted import (#7397).
import pkg.shapes

class Shape:
    def __init__(self) -> None:
        self.edges: int = 3
    def total(self) -> int:
        return self.edges

b = pkg.shapes.Shape()
assert b.total() == 4
