import numpy as np

a = np.array([1.9, 2.9], dtype=np.int32)
v = a[1:]
assert v[0] == 2.9
