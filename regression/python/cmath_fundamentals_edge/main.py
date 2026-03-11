import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


z0 = complex(0.0, 0.0)
p0 = cmath.phase(z0)
assert approx(p0, 0.0, 1e-6)

p_neg0 = cmath.phase(complex(0.0, -0.0))
phase_neg0_sample = p_neg0

sq0 = cmath.sqrt(z0)
assert approx(sq0.real, 0.0, 1e-6)
assert approx(sq0.imag, 0.0, 1e-6)

l1 = cmath.log(complex(1.0, 0.0))
assert approx(l1.real, 0.0, 1e-6)
assert approx(l1.imag, 0.0, 1e-6)

l2 = cmath.log(complex(2.0, 0.0))
assert l2.real > 0.68 and l2.real < 0.70
assert approx(l2.imag, 0.0, 1e-6)

l2_base = cmath.log(complex(2.0, 0.0), complex(2.0, 0.0))
assert approx(l2_base.real, 1.0, 1e-5)
assert approx(l2_base.imag, 0.0, 1e-5)

raised = False
try:
    cmath.log(complex(2.0, 0.0), foo=complex(2.0, 0.0))
except TypeError:
    raised = True
assert raised

raised = False
try:
    cmath.log(complex(2.0, 0.0), base=complex(2.0, 0.0))
except TypeError:
    raised = True
assert raised

raised = False
try:
    cmath.log10(complex(2.0, 0.0), base=complex(10.0, 0.0))
except TypeError:
    raised = True
assert raised

raised = False
try:
    cmath.log()
except TypeError:
    raised = True
assert raised

raised = False
try:
    cmath.log10()
except TypeError:
    raised = True
assert raised

raised = False
try:
    cmath.log10(complex(2.0, 0.0), complex(10.0, 0.0))
except TypeError:
    raised = True
assert raised

raised = False
try:
    cmath.log(complex(2.0, 0.0), complex(0.0, 0.0))
except ValueError:
    raised = True
if not raised:
    log_div_zero_sample = cmath.log(complex(2.0, 0.0), complex(0.0, 0.0))

ln_neg = cmath.log(complex(-2.0, 0.0))
assert ln_neg.real > 0.68 and ln_neg.real < 0.70
assert ln_neg.imag > 3.1 and ln_neg.imag < 3.2

raised = False
try:
    cmath.log(complex(-0.0, -0.0))
except ValueError:
    raised = True
if not raised:
    log_neg_zero_sample = cmath.log(complex(-0.0, -0.0))

sq_neg_zero = cmath.sqrt(complex(-0.0, -0.0))
sqrt_neg_zero_real = sq_neg_zero.real
sqrt_neg_zero_imag = sq_neg_zero.imag

rp = cmath.polar(complex(1.0, 0.0))
polar_r = rp[0]
polar_phi = rp[1]
