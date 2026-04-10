# repr() should produce the same output as str() for complex numbers.

z1 = complex(1, 2)
r1 = repr(z1)
assert r1 == "(1+2j)"

z2 = complex(0, -5)
r2 = repr(z2)
assert r2 == "-5j"

z3 = complex(-3, 0)
r3 = repr(z3)
assert r3 == "(-3+0j)"
