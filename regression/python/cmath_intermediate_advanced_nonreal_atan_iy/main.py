import cmath
import math

def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x

def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol

y = 0.2
iy = complex(0.0, y)
r = cmath.atan(iy)
assert approx(r.real, 0.0, 1e-4)
assert approx(r.imag, math.atanh(y), 1e-4)
