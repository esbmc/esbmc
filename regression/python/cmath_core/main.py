import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


z = complex(1.0, 2.0)

ph = cmath.phase(z)
assert ph > 1.0 and ph < 1.2

assert cmath.isfinite(z)
assert not cmath.isnan(z)
assert not cmath.isinf(z)
assert cmath.isnan(cmath.nanj)
assert cmath.isinf(cmath.infj)

assert cmath.isclose(z, z + complex(1e-12, -1e-12), rel_tol=1e-9, abs_tol=1e-9)

# constants sanity
assert approx(cmath.pi, math.pi)
assert approx(cmath.e, math.e)
assert approx(cmath.tau, math.tau)
