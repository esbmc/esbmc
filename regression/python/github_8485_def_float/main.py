# A function passed to a parameter keeps its float return (#8485).
def half(a: int) -> float:
    return a / 2


def apply(g, v):
    return g(v)


assert apply(half, 3) == 1.5
