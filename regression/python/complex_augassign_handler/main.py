# Augmented assignment operations that exercise the handler's
# binary op consolidation paths.

# += with complex.
z = complex(1, 2)
z += complex(3, 4)
assert z == complex(4, 6)

# += with int (promotion path).
z2 = complex(1, 2)
z2 += 5
assert z2 == complex(6, 2)

# += with float.
z3 = complex(1, 2)
z3 += 3.5
assert z3 == complex(4.5, 2)

# -= with complex.
z4 = complex(10, 8)
z4 -= complex(3, 2)
assert z4 == complex(7, 6)

# *= with complex.
z5 = complex(1, 1)
z5 *= complex(1, 1)
# (1+1j)*(1+1j) = 2j
assert abs(z5.real) < 1e-10
assert abs(z5.imag - 2.0) < 1e-10

# *= with int.
z6 = complex(3, 4)
z6 *= 2
assert z6 == complex(6, 8)

# /= with complex.
z7 = complex(3, 1)
z7 /= complex(1, 0)
assert z7 == complex(3, 1)

# /= with int.
z8 = complex(6, 4)
z8 /= 2
assert z8 == complex(3, 2)

# += with bool.
z9 = complex(1, 2)
z9 += True
assert z9 == complex(2, 2)

# *= with False gives zero.
z10 = complex(5, 3)
z10 *= False
assert z10.real == 0.0
assert z10.imag == 0.0
