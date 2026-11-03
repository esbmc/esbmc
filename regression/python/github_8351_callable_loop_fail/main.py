class Line:
    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, t: int) -> int:
        return self.k + t


def total(lines: list[Line]) -> int:
    s = 0
    for f in lines:
        p = f(3)
        s += p
    return s


assert total([Line(1), Line(2)]) == 10
