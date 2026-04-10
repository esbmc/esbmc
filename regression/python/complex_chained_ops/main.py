import math

# Chained additions.
z = complex(1, 1) + complex(2, 3) + complex(4, 5)
assert z == complex(7, 9)

# Chained multiplications.
w = complex(1, 0) * complex(0, 1) * complex(0, 1)
# j * j = -1, so 1 * j * j = -1+0j
assert abs(w.real - (-1.0)) < 1e-10
assert abs(w.imag) < 1e-10

# Mixed chain: add then mul.
r = (complex(1, 2) + complex(3, 4)) * complex(0, 1)
# (4+6j) * j = -6+4j
assert abs(r.real - (-6.0)) < 1e-10
assert abs(r.imag - 4.0) < 1e-10

# Division chain: left-to-right associativity.
d = complex(8, 0) / complex(2, 0) / complex(2, 0)
assert abs(d.real - 2.0) < 1e-10
assert abs(d.imag) < 1e-10

# Triple negation.
z1 = complex(3, 4)
assert -(-(-z1)) == complex(-3, -4)

# Negation of sum.
z2 = complex(1, 2) + complex(3, 4)
assert -(z2) == complex(-4, -6)

# Conjugate then abs.
z3 = complex(3, 4)
assert abs(z3.conjugate()) == 5.0

# Mul then compare.
z4 = complex(2, 0)
z5 = complex(3, 0)
assert z4 * z5 == complex(6, 0)
assert z4 * z5 == 6

# Sub then div.
z6 = (complex(10, 0) - complex(6, 0)) / complex(2, 0)
assert z6 == complex(2, 0)
