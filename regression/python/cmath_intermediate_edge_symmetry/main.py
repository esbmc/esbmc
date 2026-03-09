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
assert cmath.isfinite(cmath.sinh(z))
assert cmath.isfinite(cmath.cosh(z))
assert cmath.isfinite(cmath.tanh(z))

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

sh_par = cmath.sinh(z_par)
sh_neg = cmath.sinh(z_neg)
assert approx(sh_neg.real, -sh_par.real, 1e-5)
assert approx(sh_neg.imag, -sh_par.imag, 1e-5)

ch_par = cmath.cosh(z_par)
ch_neg = cmath.cosh(z_neg)
assert approx(ch_neg.real, ch_par.real, 1e-5)
assert approx(ch_neg.imag, ch_par.imag, 1e-5)

t_par = cmath.tan(z_par)
t_neg = cmath.tan(z_neg)
assert approx(t_neg.real, -t_par.real, 1e-5)
assert approx(t_neg.imag, -t_par.imag, 1e-5)

th_par = cmath.tanh(z_par)
th_neg = cmath.tanh(z_neg)
assert approx(th_neg.real, -th_par.real, 1e-5)
assert approx(th_neg.imag, -th_par.imag, 1e-5)
