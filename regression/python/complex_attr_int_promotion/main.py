# Tests .real and .imag on integer-constructed complex values
# Verifies that int args are properly promoted to float

z1 = complex(3, 4)
assert z1.real == 3.0
assert z1.imag == 4.0

# Single int argument
z2 = complex(7, 0)
assert z2.real == 7.0
assert z2.imag == 0.0

# Negative integers
z3 = complex(-1, -2)
assert z3.real == -1.0
assert z3.imag == -2.0

# Zero integer args
z4 = complex(0, 0)
assert z4.real == 0.0
assert z4.imag == 0.0

# Mixed int and float
z5 = complex(3, 4.5)
assert z5.real == 3.0
assert z5.imag == 4.5

# Bool args (promoted to int then to float)
z6 = complex(True, False)
assert z6.real == 1.0
assert z6.imag == 0.0
