# A class tag is keyed by name alone, so the local Shape never got its own
# symbol: every use bound to the imported one and `s.total()` read `sides`,
# reporting VERIFICATION FAILED on an assertion CPython holds (#7397).
import shapes


class Shape:
    def __init__(self) -> None:
        self.edges: int = 3

    def total(self) -> int:
        return self.edges


s = Shape()
assert s.total() == 3
