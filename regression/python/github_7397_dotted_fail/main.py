# pkg.shapes.Shape keeps its own total() (#7397).
import pkg.shapes

class Shape:
    def __init__(self) -> None:
        self.edges: int = 3
    def total(self) -> int:
        return self.edges

b = pkg.shapes.Shape()
assert b.total() == 3
