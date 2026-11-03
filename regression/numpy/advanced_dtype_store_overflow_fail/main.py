import numpy as np

a = np.array([1], dtype=np.int8)
n = 300
a[0] = n
assert a[0] == 44
