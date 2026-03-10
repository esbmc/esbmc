import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol


two_pi = 2.0 * math.pi
z_per = complex(0.4, 0.2)
s0 = cmath.sin(z_per)
s1 = cmath.sin(z_per + complex(two_pi, 0.0))
c0 = cmath.cos(z_per)
c1 = cmath.cos(z_per + complex(two_pi, 0.0))
t0 = cmath.tan(z_per)
t1 = cmath.tan(z_per + complex(two_pi, 0.0))

assert approx(s0.real, s1.real, 1e-5)
assert approx(s0.imag, s1.imag, 1e-5)
assert approx(c0.real, c1.real, 1e-5)
assert approx(c0.imag, c1.imag, 1e-5)
assert approx(t0.real, t1.real, 1e-5)
assert approx(t0.imag, t1.imag, 1e-5)
