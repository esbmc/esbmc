# The imported module's Shape used to share the local Shape's symbol; the
# parser now converts it as Shape$shapes, so the local class keeps its own
# methods (#7397).
import shapes


class Shape:
    def __init__(self) -> None:
        self.edges: int = 3

    def total(self) -> int:
        return self.edges


s = Shape()
assert s.total() == 3
