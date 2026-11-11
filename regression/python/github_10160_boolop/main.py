# An in-place list method is None as an and/or operand (#10160).
xs = [1]
v = xs.append(0) or 2
assert v == 2
assert len(xs) == 2

w = xs.clear() or 0
assert w == 0
assert len(xs) == 0

u = xs.append(5) and 2
assert u is None
assert len(xs) == 1

k = 0
if xs.append(0) and k == 0:
    k = 5
assert k == 0
assert len(xs) == 2
