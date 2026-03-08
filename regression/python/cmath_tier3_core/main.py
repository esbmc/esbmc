import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


z = complex(1.0, 1.0)

assert approx(cmath.pi, math.pi)
assert approx(cmath.e, math.e)
assert approx(cmath.tau, math.tau)

assert cmath.isfinite(complex(1.0, 0.0))
assert not cmath.isfinite(cmath.infj)
assert cmath.isinf(cmath.infj)
assert not cmath.isnan(cmath.infj)
assert cmath.isnan(cmath.nanj)
assert not cmath.isinf(cmath.nanj)

assert cmath.isclose(complex(-0.0, 0.0), complex(0.0, -0.0))
assert cmath.isclose(cmath.infj, cmath.infj)
assert cmath.isclose(z, z + complex(1e-12, -1e-12), rel_tol=1e-9, abs_tol=1e-9)
