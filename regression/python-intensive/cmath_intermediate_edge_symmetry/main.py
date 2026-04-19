import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol


z = complex(2.0, 3.0)
assert cmath.isfinite(cmath.sin(z))
assert cmath.isfinite(cmath.cos(z))
assert cmath.isfinite(cmath.tan(z))

z_par = complex(0.7, -0.4)
z_neg = complex(-0.7, 0.4)

s_par = cmath.sin(z_par)
s_neg = cmath.sin(z_neg)
assert approx(s_neg.real, -s_par.real, 1e-5)
assert approx(s_neg.imag, -s_par.imag, 1e-5)

c_par = cmath.cos(z_par)
c_neg = cmath.cos(z_neg)
assert approx(c_neg.real, c_par.real, 1e-5)
assert approx(c_neg.imag, c_par.imag, 1e-5)

t_par = cmath.tan(z_par)
t_neg = cmath.tan(z_neg)
assert approx(t_neg.real, -t_par.real, 1e-5)
assert approx(t_neg.imag, -t_par.imag, 1e-5)
