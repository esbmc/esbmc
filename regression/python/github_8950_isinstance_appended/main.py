# isinstance on a list element added by append or a comprehension (#8950).
xs = []
xs.append(0)
assert isinstance(xs[0], int)
assert isinstance(xs[-1], int)
assert not isinstance(xs[0], float)
assert not isinstance(xs[0], str)

ys = [i for i in range(2)]
assert isinstance(ys[0], int)

zs = [1, 2.5]
k = 1
assert isinstance(zs[k], float)
assert not isinstance(zs[k], int)

bs = []
bs.append(True)
assert isinstance(bs[0], bool)
assert isinstance(bs[0], int)

ns = [3]
assert isinstance(ns[k - 1], int)
d = {"a": "x"}
assert isinstance(d["a"], str)
