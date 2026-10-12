class V:
    def __init__(self, x: int):
        self.x = x

    def scale(self, k: int) -> int:
        return self.x * k


v = V(2)
assert getattr(v, "scale")(3) == 7
