# str() with float components and trailing zero stripping.

z1 = complex(1.5, 2.5)
s1 = str(z1)
assert s1 == "(1.5+2.5j)"

z2 = complex(3.0, 4.0)
s2 = str(z2)
assert s2 == "(3+4j)"

# Negative real and positive imaginary.
z3 = complex(-1, 2)
s3 = str(z3)
assert s3 == "(-1+2j)"

# Negative both.
z4 = complex(-1, -2)
s4 = str(z4)
assert s4 == "(-1-2j)"
