import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol


x = 0.3
sin_real = cmath.sin(complex(x, 0.0))
cos_real = cmath.cos(complex(x, 0.0))
sinh_real = cmath.sinh(complex(x, 0.0))
cosh_real = cmath.cosh(complex(x, 0.0))

assert approx(sin_real.real, math.sin(x), 1e-5)
assert approx(sin_real.imag, 0.0, 1e-5)
assert approx(cos_real.real, math.cos(x), 1e-5)
assert approx(cos_real.imag, 0.0, 1e-5)
assert approx(sinh_real.real, math.sinh(x), 1e-5)
assert approx(sinh_real.imag, 0.0, 1e-5)
assert approx(cosh_real.real, math.cosh(x), 1e-5)
assert approx(cosh_real.imag, 0.0, 1e-5)

atanh_hi = cmath.atanh(complex(0.9999, 0.0))
atanh_lo = cmath.atanh(complex(-0.9999, 0.0))
assert atanh_hi.real > 4.0
assert approx(atanh_hi.imag, 0.0, 1e-5)
assert atanh_lo.real < -4.0
assert approx(atanh_lo.imag, 0.0, 1e-5)

asin_pos1 = cmath.asin(complex(1.0, 0.0))
asin_neg1 = cmath.asin(complex(-1.0, 0.0))
acos_pos1 = cmath.acos(complex(1.0, 0.0))
acos_neg1 = cmath.acos(complex(-1.0, 0.0))
atan_pos1 = cmath.atan(complex(1.0, 0.0))
atan_neg1 = cmath.atan(complex(-1.0, 0.0))

assert approx(asin_pos1.real, math.pi / 2.0, 1e-5)
assert approx(asin_pos1.imag, 0.0, 1e-5)
assert approx(asin_neg1.real, -math.pi / 2.0, 1e-5)
assert approx(asin_neg1.imag, 0.0, 1e-5)
assert approx(acos_pos1.real, 0.0, 1e-5)
assert approx(acos_pos1.imag, 0.0, 1e-5)
assert approx(acos_neg1.real, math.pi, 1e-5)
assert approx(acos_neg1.imag, 0.0, 1e-5)
assert approx(atan_pos1.real, math.pi / 4.0, 1e-5)
assert approx(atan_pos1.imag, 0.0, 1e-5)
assert approx(atan_neg1.real, -math.pi / 4.0, 1e-5)
assert approx(atan_neg1.imag, 0.0, 1e-5)
