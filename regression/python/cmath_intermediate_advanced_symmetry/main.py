import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol


x = 0.25
asin_r = cmath.asin(complex(x, 0.0))
acos_r = cmath.acos(complex(x, 0.0))
atan_r = cmath.atan(complex(x, 0.0))
asinh_r = cmath.asinh(complex(x, 0.0))
atanh_r = cmath.atanh(complex(x, 0.0))

asin_neg_r = cmath.asin(complex(-x, 0.0))
acos_neg_r = cmath.acos(complex(-x, 0.0))
atan_neg_r = cmath.atan(complex(-x, 0.0))
asinh_neg_r = cmath.asinh(complex(-x, 0.0))
atanh_neg_r = cmath.atanh(complex(-x, 0.0))

assert approx(asin_neg_r.real, -asin_r.real, 1e-4)
assert approx(asin_neg_r.imag, -asin_r.imag, 1e-4)
assert approx(acos_neg_r.real, math.pi - acos_r.real, 1e-4)
assert approx(acos_neg_r.imag, -acos_r.imag, 1e-4)
assert approx(atan_neg_r.real, -atan_r.real, 1e-4)
assert approx(atan_neg_r.imag, -atan_r.imag, 1e-4)
assert approx(asinh_neg_r.real, -asinh_r.real, 1e-4)
assert approx(asinh_neg_r.imag, -asinh_r.imag, 1e-4)
assert approx(atanh_neg_r.real, -atanh_r.real, 1e-4)
assert approx(atanh_neg_r.imag, -atanh_r.imag, 1e-4)

sum_asin_acos = asin_r + acos_r
assert approx(sum_asin_acos.real, math.pi / 2.0, 1e-4)
assert approx(sum_asin_acos.imag, 0.0, 1e-4)

sum_acos = acos_r + acos_neg_r
assert approx(sum_acos.real, math.pi, 1e-4)
assert approx(sum_acos.imag, 0.0, 1e-4)

sum_asin = asin_r + asin_neg_r
assert approx(sum_asin.real, 0.0, 1e-4)
assert approx(sum_asin.imag, 0.0, 1e-4)
