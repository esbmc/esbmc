import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def assert_close(a: float, b: float, tol: float = 1e-6) -> None:
    assert absf(a - b) <= tol


assert_close(math.fmod(5.5, 2.0), 1.5, 1e-6)
