# Issue #8950: isinstance on an element added by append or a comprehension.
xs = []
xs.append(0)
assert isinstance(xs[0], int)
assert not isinstance(xs[0], str)

ys = [i for i in range(2)]
assert isinstance(ys[1], int)

zs = []
if nondet_bool():
    zs.append(1)
else:
    zs.append("a")
assert isinstance(zs[0], int) or isinstance(zs[0], str)
