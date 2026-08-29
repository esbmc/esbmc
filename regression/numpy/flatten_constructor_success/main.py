import numpy as np

# np.flatten(a), module-function form, with a built by a shape constructor.
a = np.identity(2)
b = np.flatten(a)
assert b[0] == 1
assert b[1] == 0
assert b[2] == 0
assert b[3] == 1

# a.flatten(), method form, with a built by a different shape constructor.
c = np.zeros((2, 2))
d = c.flatten()
assert d[0] == 0
assert d[1] == 0
assert d[2] == 0
assert d[3] == 0
