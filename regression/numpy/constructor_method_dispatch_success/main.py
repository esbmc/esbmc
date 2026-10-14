import numpy as np

# transpose: module vs method form, both on an eye()-built 2D array.
a = np.eye(2)
ta = np.transpose(a)
tb = a.transpose()
assert ta[0][1] == tb[0][1]
assert ta[1][0] == tb[1][0]

# flatten: module vs method form, both on an identity()-built array.
b = np.identity(2)
fa = np.flatten(b)
fb = b.flatten()
assert fa[0] == fb[0]
assert fa[3] == fb[3]

# sum/mean: module vs method form compared directly in the same assert
# (no intermediate variable), on a full()-built array.
c = np.full((2, 2), 3)
assert np.sum(c) == c.sum()
assert np.mean(c) == c.mean()

# min/max/var: module vs method form compared directly in the same assert,
# on an eye()-built array.
d = np.eye(3)
assert np.min(d) == d.min()
assert np.max(d) == d.max()
assert np.var(d) == d.var()
