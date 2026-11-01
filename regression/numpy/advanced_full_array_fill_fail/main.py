import numpy as np

b = np.array([300, 2])
a = np.full((1, 2), b, dtype=np.int8)
assert a[0][0] == 300
