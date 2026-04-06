# Tests .real and .imag on results of arithmetic operations

# After addition
z1 = complex(1.0, 2.0) + complex(3.0, 4.0)
assert z1.real == 4.0
assert z1.imag == 6.0

# After subtraction
z2 = complex(5.0, 10.0) - complex(2.0, 3.0)
assert z2.real == 3.0
assert z2.imag == 7.0

# After multiplication
z3 = complex(1.0, 2.0) * complex(3.0, 4.0)
# (1+2j)*(3+4j) = 3+4j+6j+8j^2 = (3-8)+(4+6)j = -5+10j
assert z3.real == -5.0
assert z3.imag == 10.0

# After negation
z4_base = complex(3.0, 4.0)
z4 = -z4_base
assert z4.real == -3.0
assert z4.imag == -4.0

# After conjugate, access components
z5 = complex(3.0, 4.0)
c5 = z5.conjugate()
assert c5.real == 3.0
assert c5.imag == -4.0
