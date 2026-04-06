import cmath
import math

# Basic cmath.log of a positive real complex number.
z1 = complex(1.0, 0.0)
w1 = cmath.log(z1)
assert w1.real == 0.0
assert w1.imag == 0.0

# cmath.log of e should give ~1+0j.
z2 = complex(math.e, 0.0)
w2 = cmath.log(z2)
assert abs(w2.real - 1.0) < 1e-6
assert abs(w2.imag) < 1e-6

# cmath.log of a negative real: ln(-1) = pi*j.
z3 = complex(-1.0, 0.0)
w3 = cmath.log(z3)
assert abs(w3.real) < 1e-6
assert abs(w3.imag - math.pi) < 1e-6

# cmath.log of a purely imaginary number: ln(j) = pi/2 * j.
z4 = complex(0.0, 1.0)
w4 = cmath.log(z4)
assert abs(w4.real) < 1e-6
assert abs(w4.imag - math.pi / 2.0) < 1e-6

# cmath.log10 basic: log10(10+0j) == 1+0j.
z5 = complex(10.0, 0.0)
w5 = cmath.log10(z5)
assert abs(w5.real - 1.0) < 1e-6
assert abs(w5.imag) < 1e-6

# cmath.log10 of 1 should give 0+0j.
z6 = complex(1.0, 0.0)
w6 = cmath.log10(z6)
assert w6.real == 0.0
assert w6.imag == 0.0

# cmath.log with base argument: log(100, 10) == 2+0j.
z7 = complex(100.0, 0.0)
b7 = complex(10.0, 0.0)
w7 = cmath.log(z7, 10.0)
assert abs(w7.real - 2.0) < 1e-5
assert abs(w7.imag) < 1e-5
