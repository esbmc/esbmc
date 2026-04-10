import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


assert approx(cmath.pi, math.pi)
assert approx(cmath.e, math.e)
assert approx(cmath.tau, math.tau)
assert math.isinf(cmath.inf)
assert math.isnan(cmath.nan)

assert cmath.isfinite(complex(1.0, 0.0))
assert not cmath.isfinite(cmath.infj)
assert cmath.isinf(cmath.infj)
assert not cmath.isnan(cmath.infj)
assert cmath.isnan(cmath.nanj)
assert not cmath.isinf(cmath.nanj)

assert cmath.isclose(complex(-0.0, 0.0), complex(0.0, -0.0))

# complex power basics from Tier 3 scope
z = complex(1.0, 2.0)
z2 = z ** 2
assert approx(z2.real, -3.0, 1e-6)
assert approx(z2.imag, 4.0, 1e-6)

zsqrt = complex(4.0, 0.0) ** 0.5
assert approx(zsqrt.real, 2.0, 1e-6)
assert approx(zsqrt.imag, 0.0, 1e-6)

zsqrt_c = complex(4.0, 0.0) ** complex(0.5, 0.0)
assert approx(zsqrt_c.real, 2.0, 1e-6)
assert approx(zsqrt_c.imag, 0.0, 1e-6)
