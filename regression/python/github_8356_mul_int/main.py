class V:
    def __init__(self, x: int):
        self.x = x

    def __mul__(self, k: int) -> "V":
        return V(self.x * k)

    def __rmul__(self, k: int) -> "V":
        return V(self.x * k + 1)

k = 3
assert (V(2) * 3).x == 6
assert (V(2) * k).x == 6
assert (3 * V(2)).x == 7
assert (k * V(2)).x == 7
