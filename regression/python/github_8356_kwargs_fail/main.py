class V:
    def __init__(self, x: int):
        self.x = x

    def scale(self, k: int, offset: int = 1, *, sign: int = 1) -> int:
        return sign * (self.x * k + offset)

    times = scale

v = V(2)
assert v.times(3, offset=4) == 6
