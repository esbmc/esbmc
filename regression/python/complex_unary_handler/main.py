import math

# Unary minus negates both components.
z = complex(3.0, 4.0)
neg_z = -z
assert neg_z == complex(-3.0, -4.0)

# Unary plus is identity.
pos_z = +z
assert pos_z == z

# Double negation.
assert -(-z) == z

# Negation then plus.
assert +(-(z)) == complex(-3.0, -4.0)

# Plus then negation.
assert -(+z) == complex(-3.0, -4.0)

# Unary minus on purely real complex.
z2 = complex(5.0, 0.0)
assert -z2 == complex(-5.0, 0.0)

# Unary minus on purely imaginary.
z3 = complex(0.0, 7.0)
assert -z3 == complex(0.0, -7.0)

# Unary minus on zero complex.
z4 = complex(0.0, 0.0)
neg_z4 = -z4
assert neg_z4.real == 0.0
assert neg_z4.imag == 0.0

# Unary operations preserve magnitude.
z5 = complex(3.0, 4.0)
assert abs(-z5) == abs(z5)
assert abs(+z5) == abs(z5)

# Signed-zero behavior: -(-0.0, 0.0) -> (0.0, -0.0).
z6 = complex(-0.0, 0.0)
neg_z6 = -z6
assert neg_z6.real == 0.0
assert neg_z6.imag == 0.0
