import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


z = complex(0.25, -0.5)

s = cmath.sin(z)
c = cmath.cos(z)
t = cmath.tan(z)
ratio = s / c
assert approx(t.real, ratio.real, 1e-5)
assert approx(t.imag, ratio.imag, 1e-5)

sh = cmath.sinh(z)
ch = cmath.cosh(z)
th = cmath.tanh(z)
ratio_h = sh / ch
assert approx(th.real, ratio_h.real, 1e-5)
assert approx(th.imag, ratio_h.imag, 1e-5)

z0 = complex(0.0, 0.0)
asin0 = cmath.asin(z0)
acos0 = cmath.acos(z0)
atan0 = cmath.atan(z0)
asinh0 = cmath.asinh(z0)
atanh0 = cmath.atanh(z0)

assert approx(asin0.real, 0.0)
assert approx(asin0.imag, 0.0)
assert approx(acos0.real, cmath.pi / 2.0, 1e-6)
assert approx(acos0.imag, 0.0)
assert approx(atan0.real, 0.0)
assert approx(atan0.imag, 0.0)
assert approx(asinh0.real, 0.0)
assert approx(asinh0.imag, 0.0)
assert approx(atanh0.real, 0.0)
assert approx(atanh0.imag, 0.0)

acosh1 = cmath.acosh(complex(1.0, 0.0))
assert approx(acosh1.real, 0.0)
assert approx(acosh1.imag, 0.0)

# Tier-2 addendum in plan: log10(z) == log(z) / log(10)
z_log = complex(2.0, 0.5)
lz10 = cmath.log10(z_log)
lz_div = cmath.log(z_log) / cmath.log(complex(10.0, 0.0))
assert approx(lz10.real, lz_div.real, 1e-5)
assert approx(lz10.imag, lz_div.imag, 1e-5)

# principal branch on the negative real axis: log10(-1) = i*pi/ln(10)
lneg1 = cmath.log10(complex(-1.0, 0.0))
assert approx(lneg1.real, 0.0, 1e-5)
assert approx(lneg1.imag, cmath.pi / math.log(10.0), 1e-5)
