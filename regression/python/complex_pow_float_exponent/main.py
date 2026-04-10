import math

# Float exponent on complex goes through exp(exponent * log(z)) path.

# (4+0j) ** 0.5 = sqrt(4) = 2+0j
z1 = complex(4, 0)
w1 = z1 ** 0.5
assert abs(w1.real - 2.0) < 1e-5
assert abs(w1.imag) < 1e-5

# (4+0j) ** -0.5 = 1/sqrt(4) = 0.5+0j
z2 = complex(4, 0)
w2 = z2 ** (-0.5)
assert abs(w2.real - 0.5) < 1e-5
assert abs(w2.imag) < 1e-5
