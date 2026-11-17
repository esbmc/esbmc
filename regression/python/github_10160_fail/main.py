# Issue #10160: an in-place list method assigned to a name gives None.
xs = [1]
r = xs.append(0)
assert r is not None
