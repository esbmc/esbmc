import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def assert_close(a: float, b: float, tol: float = 1e-6) -> None:
    assert absf(a - b) <= tol


math.atanh(2.0)
assert False
