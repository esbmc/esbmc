import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


# constants and predicates for non-finite values
assert cmath.isnan(cmath.nanj)
assert cmath.isinf(cmath.infj)
assert not cmath.isfinite(cmath.infj)
assert not cmath.isfinite(cmath.nanj)

# isclose edge behavior
assert cmath.isclose(complex(1.0, 1.0), complex(1.0, 1.0))
assert cmath.isclose(complex(-0.0, 0.0), complex(0.0, -0.0))
assert cmath.isclose(complex(1.0, 0.0), complex(1.0 + 1e-12, 0.0), abs_tol=1e-9)

raised = False
try:
    cmath.isclose(complex(1.0, 0.0), complex(1.0, 0.0), abs_tol=-1.0)
except ValueError:
    raised = True
assert raised

assert approx(cmath.pi, math.pi)
assert approx(cmath.tau, 2.0 * math.pi)
