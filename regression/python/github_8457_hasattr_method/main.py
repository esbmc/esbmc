class Base:
    def describe(self) -> int:
        return 0


class V(Base):
    def __init__(self, x: int):
        self.x = x

    def scale(self, k: int) -> int:
        return self.x * k


v = V(2)
assert hasattr(v, "scale")
assert hasattr(v, "describe")
assert hasattr(v, "x")
assert not hasattr(v, "missing")
assert not hasattr(v, "V")
