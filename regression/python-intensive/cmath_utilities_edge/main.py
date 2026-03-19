import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


# polar/rect edge behavior
p0 = cmath.polar(complex(0.0, 0.0))
assert approx(p0[0], 0.0)
assert approx(p0[1], 0.0)

r0 = cmath.rect(0.0, 1.234)
assert approx(r0.real, 0.0)
assert approx(r0.imag, 0.0)

# non-finite predicates with mixed components
assert cmath.isinf(complex(math.inf, 1.0))
assert cmath.isinf(complex(1.0, -math.inf))
assert cmath.isnan(complex(math.nan, 1.0))
assert cmath.isnan(complex(1.0, math.nan))
assert not cmath.isfinite(complex(math.inf, 1.0))
assert not cmath.isfinite(complex(1.0, math.nan))

# isclose edge behavior
assert not cmath.isclose(complex(math.nan, 0.0), complex(0.0, 0.0))
assert cmath.isclose(complex(math.inf, 0.0), complex(math.inf, 0.0))
assert cmath.isclose(complex(0.0, 0.0), complex(1e-12, -1e-12), abs_tol=1e-9)

raised = False
try:
    cmath.isclose(complex(1.0, 0.0), complex(1.0, 0.0), rel_tol=-1.0)
except ValueError:
    raised = True
assert raised

# complex power identity edges
z = complex(1.0, -2.0)
assert z ** 0 == complex(1.0, 0.0)
assert z ** 1 == z
assert z ** False == complex(1.0, 0.0)
assert z ** True == z

z3 = z ** 3
assert approx(z3.real, -11.0)
assert approx(z3.imag, 2.0)

n = +3
z3_alias = z ** n
assert approx(z3_alias.real, -11.0)
assert approx(z3_alias.imag, 2.0)

zinv = z ** -1
assert approx(zinv.real, 0.2)
assert approx(zinv.imag, 0.4)

z8 = complex(1.0, 1.0) ** 8
assert approx(z8.real, 16.0)
assert approx(z8.imag, 0.0)

raised_pow = False
try:
    complex(1.0, 2.0) ** "x"
except TypeError:
    raised_pow = True
assert raised_pow
