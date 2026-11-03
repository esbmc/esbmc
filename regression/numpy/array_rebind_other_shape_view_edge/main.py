import numpy as np

a = np.array([1, 2, 3, 4])
v = a[1:3]
a = np.zeros(2)
v[0] = 9
assert v[0] == 9 and v[1] == 3
assert a[1] == 0.0
assert len(a) == 2
