class P:
    def __init__(self, x: int):
        self.x = x


def f(c: bool) -> int:
    if c:
        b = [1, 2]
        p = P(3)
    return b[0] + p.x


assert f(True) == 4
