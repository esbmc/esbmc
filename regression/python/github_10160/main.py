# Issue #10160: an in-place list method read as a value is None.
xs = [1]
r = xs.append(0)
assert r is None
v = xs.append(0) or 2
assert v == 2
assert len(xs) == 3
w = xs.clear() or 0
assert w == 0
assert len(xs) == 0
