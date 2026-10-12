class V:
    def __init__(self, x: int):
        self.x = x

    def __mul__(self, k: int) -> "V":
        return V(self.x * k)

    __rmul__ = __mul__

v = 3 * V(2)
assert v.x == 6
