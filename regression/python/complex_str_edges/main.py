# str() on complex with negative zero and edge values.

# Pure real (positive).
z1 = complex(5, 0)
s1 = str(z1)
assert s1 == "(5+0j)"

# Pure real (negative).
z2 = complex(-3, 0)
s2 = str(z2)
assert s2 == "(-3+0j)"

# Pure imaginary unit.
z3 = complex(0, 1)
s3 = str(z3)
assert s3 == "1j"

# Zero.
z4 = complex(0, 0)
s4 = str(z4)
assert s4 == "0j"
