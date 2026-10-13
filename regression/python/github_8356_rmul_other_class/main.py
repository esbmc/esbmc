class W:
    pass


class V:
    def __init__(self, x: int):
        self.x = x

    def __rmul__(self, k: int) -> int:
        return self.x


assert W() * V(2) == 2
