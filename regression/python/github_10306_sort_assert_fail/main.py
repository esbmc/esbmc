class C:

    def __init__(self, x: int) -> None:
        self.x = x

    def __lt__(self, other: "C") -> bool:
        assert self.x != 0
        return self.x < other.x


xs = [C(1), C(0)]
xs.sort()
