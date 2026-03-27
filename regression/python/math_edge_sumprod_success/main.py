import math



def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def assert_close(a: float, b: float, tol: float = 1e-6) -> None:
    assert absf(a - b) <= tol

assert_close(math.sumprod([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0, 1e-6)
