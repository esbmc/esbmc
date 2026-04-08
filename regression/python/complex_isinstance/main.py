# isinstance(z, complex) basic test.

z = complex(1, 2)
assert isinstance(z, complex)

# Constructed from int/float args.
z2 = complex(3, 0)
assert isinstance(z2, complex)

# Pure imaginary.
z3 = complex(0, 5)
assert isinstance(z3, complex)
