# Tests cmath.log and cmath.log10 with various complex inputs
# Focuses on log10 of complex numbers and log with 2-arg base
import cmath

# cmath.log10(10+0j) should be close to (1+0j)
w = cmath.log10(complex(10, 0))
assert w.real > 0.99 and w.real < 1.01
assert w.imag > -0.01 and w.imag < 0.01

# cmath.log(e+0j) should be close to (1+0j)
import math
z = cmath.log(complex(math.e, 0))
assert z.real > 0.99 and z.real < 1.01
assert z.imag > -0.01 and z.imag < 0.01

# cmath.log(100, 10) = log(100)/log(10) = 2
v = cmath.log(complex(100, 0), 10)
assert v.real > 1.99 and v.real < 2.01
assert v.imag > -0.01 and v.imag < 0.01


