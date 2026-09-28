import numpy as np

# np.transpose(a), module-function form, with a built by a shape constructor
# (not a np.array(...) literal).
a = np.zeros((2, 3))
b = np.transpose(a)
assert b.shape == (3, 2)
assert b[0][0] == 0
assert b[2][1] == 0

# a.transpose(), method form, with a built by a different shape constructor.
c = np.ones((2, 3))
d = c.transpose()
assert d.shape == (3, 2)
assert d[0][0] == 1
assert d[2][1] == 1
