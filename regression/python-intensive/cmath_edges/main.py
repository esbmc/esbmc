import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


# branch/edge behavior around zero
z0 = cmath.sqrt(complex(0.0, 0.0))
assert approx(z0.real, 0.0)
assert approx(z0.imag, 0.0)

# predicates
assert cmath.isfinite(complex(1.0, 0.0))
assert not cmath.isfinite(complex(math.inf, 0.0))
assert not cmath.isfinite(complex(0.0, math.nan))
assert cmath.isnan(complex(0.0, math.nan))
assert cmath.isinf(complex(math.inf, 0.0))

# isclose edge cases
assert cmath.isclose(complex(-0.0, 0.0), complex(0.0, -0.0))
assert cmath.isclose(complex(math.inf, 0.0), complex(math.inf, 0.0))

raised = False
try:
    cmath.isclose(complex(1.0, 0.0), complex(1.0, 0.0), rel_tol=-1.0)
except ValueError:
    raised = True
assert raised

# inverse/special-value edges
asin0 = cmath.asin(complex(0.0, 0.0))
assert approx(asin0.real, 0.0)
assert approx(asin0.imag, 0.0)

acos0 = cmath.acos(complex(0.0, 0.0))
assert approx(acos0.real, math.pi / 2.0, 1e-6)
assert approx(acos0.imag, 0.0)

atan0 = cmath.atan(complex(0.0, 0.0))
assert approx(atan0.real, 0.0)
assert approx(atan0.imag, 0.0)

asinh0 = cmath.asinh(complex(0.0, 0.0))
assert approx(asinh0.real, 0.0)
assert approx(asinh0.imag, 0.0)

atanh0 = cmath.atanh(complex(0.0, 0.0))
assert approx(atanh0.real, 0.0)
assert approx(atanh0.imag, 0.0)

acosh1 = cmath.acosh(complex(1.0, 0.0))
assert approx(acosh1.real, 0.0)
assert approx(acosh1.imag, 0.0)
