# str() on complex numbers with various formats.

# Mixed real and imaginary.
z1 = complex(1, 2)
s1 = str(z1)
assert s1 == "(1+2j)"

# Negative imaginary.
z2 = complex(1, -2)
s2 = str(z2)
assert s2 == "(1-2j)"

# Pure imaginary (real == 0).
z3 = complex(0, 1)
s3 = str(z3)
assert s3 == "1j"

# Pure imaginary negative.
z4 = complex(0, -3)
s4 = str(z4)
assert s4 == "-3j"

# Zero complex.
z5 = complex(0, 0)
s5 = str(z5)
assert s5 == "0j"
