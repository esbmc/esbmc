import numpy as np

a = np.array([True, False, True])
b = a.astype(int)

assert b[0] == 1
assert b[1] == 0
assert b[2] == 1

c = np.array([1, 2, 3])
d = c.astype(np.int64)
assert d[0] == 1 and d[2] == 3
assert len(d) == len(c)
d[0] = 9
assert c[0] == 1
