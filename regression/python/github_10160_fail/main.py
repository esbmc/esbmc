# Issue #10160: an in-place list method read as a value is None.
xs = [1]
v = xs.append(0) or 2
assert v == 1
