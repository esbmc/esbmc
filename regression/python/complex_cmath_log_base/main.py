import cmath
import math

# cmath.log with 2 args: log(z, base) = log(z) / log(base).

# log(100+0j, 10+0j) == 2+0j (both real-ish).
z1 = complex(100, 0)
b1 = complex(10, 0)
w1 = cmath.log(z1, b1)
assert abs(w1.real - 2.0) < 1e-5
assert abs(w1.imag) < 1e-5

# log(e+0j, e+0j) == 1+0j.
z2 = complex(math.e, 0)
w2 = cmath.log(z2, math.e)
assert abs(w2.real - 1.0) < 1e-5
assert abs(w2.imag) < 1e-5

# log(1+0j, any_base) == 0+0j.
w3 = cmath.log(complex(1, 0), 10.0)
assert w3.real == 0.0
assert w3.imag == 0.0

# log(8+0j, 2+0j) == 3+0j.
z4 = complex(8, 0)
b4 = complex(2, 0)
w4 = cmath.log(z4, b4)
assert abs(w4.real - 3.0) < 1e-4
assert abs(w4.imag) < 1e-4
