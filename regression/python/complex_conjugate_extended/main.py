# Tests conjugate on various complex values — comprehensive coverage

# Conjugate of positive components
z1 = complex(3.0, 4.0)
c1 = z1.conjugate()
assert c1 == complex(3.0, -4.0)

# Double conjugate is identity
z2 = complex(7.0, -2.0)
assert z2.conjugate().conjugate() == z2

# Conjugate of purely real is itself
z3 = complex(5.0, 0.0)
c3 = z3.conjugate()
assert c3 == z3

# Conjugate of purely imaginary negates
z4 = complex(0.0, 9.0)
c4 = z4.conjugate()
assert c4 == complex(0.0, -9.0)

# Conjugate of zero
z5 = complex(0.0, 0.0)
c5 = z5.conjugate()
assert c5 == complex(0.0, 0.0)

# Conjugate preserves real, negates imag
z6 = complex(-1.5, 2.5)
c6 = z6.conjugate()
assert c6.real == -1.5
assert c6.imag == -2.5

# z * conj(z) == |z|^2 (real result)
z7 = complex(3.0, 4.0)
product = z7 * z7.conjugate()
assert product.real == 25.0
assert product.imag == 0.0
