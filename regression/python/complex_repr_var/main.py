# repr() of a variable assigned to a complex literal.
z = complex(3, -4)
r = repr(z)
assert r == "(3-4j)"

# repr() of complex with zero imaginary.
z2 = complex(5, 0)
r2 = repr(z2)
assert r2 == "(5+0j)"

# repr() of pure imaginary.
z3 = complex(0, -7)
r3 = repr(z3)
assert r3 == "-7j"
