# repr() with wrong expected value should fail verification.
z = complex(2, 3)
r = repr(z)
assert r == "(2+4j)"
