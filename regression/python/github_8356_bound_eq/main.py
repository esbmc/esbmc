class V:
    def __init__(self, x: int):
        self.x = x

    def scale(self, k: int) -> int:
        return self.x * k

    times = scale

v = V(2)
assert v.times == v.scale
