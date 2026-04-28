import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol


z = complex(0.1, -0.2)

# direct trig/hyper values stay finite
s = cmath.sin(z)
c = cmath.cos(z)
t = cmath.tan(z)
assert cmath.isfinite(s)
assert cmath.isfinite(c)
assert cmath.isfinite(t)

sh = cmath.sinh(z)
ch = cmath.cosh(z)
th = cmath.tanh(z)
assert cmath.isfinite(sh)
assert cmath.isfinite(ch)
assert cmath.isfinite(th)

# additional edge points
z0 = complex(0.0, 0.0)
assert cmath.asin(z0) == complex(0.0, 0.0)
assert cmath.atan(z0) == complex(0.0, 0.0)
assert cmath.asinh(z0) == complex(0.0, 0.0)
assert cmath.atanh(z0) == complex(0.0, 0.0)

assert cmath.sin(z0) == complex(0.0, 0.0)
assert cmath.tan(z0) == complex(0.0, 0.0)
assert cmath.sinh(z0) == complex(0.0, 0.0)
assert cmath.tanh(z0) == complex(0.0, 0.0)

c0 = cmath.cos(z0)
assert approx(c0.real, 1.0, 1e-6)
assert approx(c0.imag, 0.0, 1e-6)

ch0 = cmath.cosh(z0)
assert approx(ch0.real, 1.0, 1e-6)
assert approx(ch0.imag, 0.0, 1e-6)

# inverse functions on real-axis non-trivial points
x = 0.25
asin_r = cmath.asin(complex(x, 0.0))
acos_r = cmath.acos(complex(x, 0.0))
atan_r = cmath.atan(complex(x, 0.0))
asinh_r = cmath.asinh(complex(x, 0.0))
atanh_r = cmath.atanh(complex(x, 0.0))
acosh_r = cmath.acosh(complex(2.0, 0.0))

assert approx(asin_r.real, math.asin(x), 1e-4)
assert approx(asin_r.imag, 0.0, 1e-4)
assert approx(acos_r.real, math.acos(x), 1e-4)
assert approx(acos_r.imag, 0.0, 1e-4)
assert approx(atan_r.real, math.atan(x), 1e-4)
assert approx(atan_r.imag, 0.0, 1e-4)
assert approx(asinh_r.real, math.asinh(x), 1e-4)
assert approx(asinh_r.imag, 0.0, 1e-4)
assert approx(atanh_r.real, math.atanh(x), 1e-4)
assert approx(atanh_r.imag, 0.0, 1e-4)
assert approx(acosh_r.real, math.acosh(2.0), 1e-4)
assert approx(acosh_r.imag, 0.0, 1e-4)
