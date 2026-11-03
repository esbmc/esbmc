def f(x: int) -> int:
    return x + 1


def g(x: float) -> float:
    return x


assert f(2) == 3
assert f(True) == 2
assert g(3) == 3.0
