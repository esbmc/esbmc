# Lambda returns int or None - assertion should fail (None is not None is False)
f = lambda n: n * 2 if n > 0 else None
x = f(-1)
assert x is not None  # Should fail: x IS None
