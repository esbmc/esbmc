a: float = 5.0
assert a.is_integer()

b: float = 5.5
assert not b.is_integer()

# Negative integral and non-integral values.
c: float = -2.0
assert c.is_integer()
d: float = -2.5
assert not d.is_integer()

# Zero is integral.
e: float = 0.0
assert e.is_integer()

# Expression receiver via the BinOp dispatch path.
f: float = 4.0
assert (f + 1.0).is_integer()
assert not (f + 0.5).is_integer()
