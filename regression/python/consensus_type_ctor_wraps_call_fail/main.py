class uint64(int):
    pass


def g(x: int) -> int:
    return x + 1


def f(x: int) -> uint64:
    return uint64(g(x))


assert f(4) == 6
