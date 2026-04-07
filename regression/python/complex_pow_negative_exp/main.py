# Tests complex power with negative integer exponents > 1
# This exercises the pow_nonnegative loop + complex_div path

z = complex(1, 1)

# z**(-2) = 1 / (z**2)
# z**2 = (1+1j)**2 = 0+2j
# 1 / (0+2j) = (0 - 0.5j)
w = z ** (-2)
assert w.real > -0.01 and w.real < 0.01     # real ≈ 0
assert w.imag > -0.51 and w.imag < -0.49    # imag ≈ -0.5

# (2+0j)**(-3) = 1/8 + 0j
w2 = complex(2, 0) ** (-3)
assert w2.real > 0.124 and w2.real < 0.126  # real ≈ 0.125
assert w2.imag > -0.01 and w2.imag < 0.01  # imag ≈ 0

# (0+1j)**(-1) = -1j  (since 1/(0+1j) = (0-1j)/(1) = -1j)
w3 = complex(0, 1) ** (-1)
assert w3.real > -0.01 and w3.real < 0.01
assert w3.imag > -1.01 and w3.imag < -0.99
