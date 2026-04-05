import cmath
import math

# TypeError: cmath.log10 with keyword args.
raised = False
try:
    cmath.log10(z=complex(1.0, 0.0))  # type: ignore
except TypeError:
    raised = True
assert raised

# TypeError: cmath.log with keyword args.
raised = False
try:
    cmath.log(z=complex(1.0, 0.0))  # type: ignore
except TypeError:
    raised = True
assert raised

# cmath.log10 of a purely imaginary number.
z1 = complex(0.0, 10.0)
w1 = cmath.log10(z1)
# log10(10j) = log10(10) + log10(j) where the magnitude is 10.
assert abs(w1.real - 1.0) < 1e-5

# cmath.log of complex with both parts nonzero.
z2 = complex(3.0, 4.0)
w2 = cmath.log(z2)
# |3+4j| = 5, so real part = ln(5)
assert abs(w2.real - math.log(5.0)) < 1e-5
# atan2(4,3) for imaginary part
assert abs(w2.imag - math.atan2(4.0, 3.0)) < 1e-5
