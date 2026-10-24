import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
c = np.copy(v)
c[0] = 70
assert v[0] == 1
assert a[1] == 1
a[2] = 40
assert c[1] == 4
assert sum(c) == 75
