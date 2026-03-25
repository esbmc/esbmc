import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


w = cmath.rect(2.0, 0.0)
assert approx(w.real, 2.0, 1e-6)
assert approx(w.imag, 0.0, 1e-6)
