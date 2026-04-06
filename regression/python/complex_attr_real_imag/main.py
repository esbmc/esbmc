# Tests .real and .imag attribute access on complex objects

# Basic access on a variable
z1 = complex(3.0, 4.0)
assert z1.real == 3.0
assert z1.imag == 4.0

# Purely real
z2 = complex(7.0, 0.0)
assert z2.real == 7.0
assert z2.imag == 0.0

# Purely imaginary
z3 = complex(0.0, -5.0)
assert z3.real == 0.0
assert z3.imag == -5.0

# Zero complex
z4 = complex(0.0, 0.0)
assert z4.real == 0.0
assert z4.imag == 0.0

# Negative components
z5 = complex(-2.5, -3.5)
assert z5.real == -2.5
assert z5.imag == -3.5

# .real and .imag in arithmetic expressions
z6 = complex(1.0, 2.0)
s = z6.real + z6.imag
assert s == 3.0

# Use .real and .imag to reconstruct the complex
z7 = complex(6.0, 8.0)
z7_copy = complex(z7.real, z7.imag)
assert z7_copy == z7

# Assign .real/.imag to variables
z8 = complex(10.0, 20.0)
r = z8.real
i = z8.imag
assert r == 10.0
assert i == 20.0
