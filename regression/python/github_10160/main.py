# Issue #10160: an in-place list method assigned to a name gives None.
xs = [2, 1]
r = xs.append(0)
assert r is None
assert len(xs) == 3
s = xs.sort()
assert s is None
assert xs[0] == 0
t = xs.remove(2)
assert t is None
u = xs.clear()
assert u is None
assert len(xs) == 0
