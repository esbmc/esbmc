class V:
    def __init__(self, x: int):
        self.x = x

    def __mul__(self, k: int) -> "V":
        return V(self.x * k)

    def __rmul__(self, k: int) -> "V":
        return V(self.x * k + 1)

k = 3
assert (k * V(2)).x == 6
