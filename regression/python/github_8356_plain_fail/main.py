class V:
    def __init__(self, x: int):
        self.x = x

    def scale(self, k: int) -> "V":
        return V(self.x * k)

    times = scale

v = V(2).times(3)
assert v.x == 5
