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

sq = cmath.sqrt(complex(4.0, 0.0))
_ = sq.real
_ = sq.imag

e0 = cmath.exp(complex(0.0, 0.0))
assert approx(e0.real, 1.0, 1e-6)
assert approx(e0.imag, 0.0, 1e-6)

l0 = cmath.log(complex(1.0, 0.0))
assert approx(l0.real, 0.0, 1e-6)
assert approx(l0.imag, 0.0, 1e-6)

l10 = cmath.log10(complex(1.0, 0.0))
assert approx(l10.real, 0.0, 1e-6)
assert approx(l10.imag, 0.0, 1e-6)

lb = cmath.log(complex(2.0, 0.0), complex(2.0, 0.0))
assert approx(lb.real, 1.0, 1e-5)
assert approx(lb.imag, 0.0, 1e-5)
