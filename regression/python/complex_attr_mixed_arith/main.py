# Tests .real/.imag on results of mixed-type arithmetic
# Verifies attribute access works after complex+float, complex+int promotions

# complex + float
z1 = complex(1.0, 2.0) + 3.0
assert z1.real == 4.0
assert z1.imag == 2.0

# complex * float
z3 = complex(2.0, 3.0) * 2.0
assert z3.real == 4.0
assert z3.imag == 6.0

# complex - float
z4 = complex(10.0, 5.0) - 3.0
assert z4.real == 7.0
assert z4.imag == 5.0

# complex / float
z5 = complex(6.0, 4.0) / 2.0
assert z5.real == 3.0
assert z5.imag == 2.0

# complex + int
z6 = complex(1.0, 2.0) + 3
assert z6.real == 4.0
assert z6.imag == 2.0
