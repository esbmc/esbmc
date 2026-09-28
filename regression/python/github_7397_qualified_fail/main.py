# shapes.Shape keeps its own total(), not the local one (#7397).
import shapes


class Shape:
    def __init__(self) -> None:
        self.edges: int = 3

    def total(self) -> int:
        return self.edges


b = shapes.Shape()
assert b.total() == 3
