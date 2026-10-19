import numpy as np

a = np.array([1.9, 2.9, 3.9], dtype=np.int32)
v = a[1:]
assert v[0] == 2
a[2] = 5
assert v[1] == 5
