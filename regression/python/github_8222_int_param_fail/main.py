def f(x: int) -> int:
    return x + 1


def g(x: float) -> float:
    return x


assert g(3) == 3.0
assert f(True) == 1
