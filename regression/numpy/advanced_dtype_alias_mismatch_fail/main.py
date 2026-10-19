import numpy as np

a = np.array([0.5], dtype=np.bool_)
b = np.array([0.5], dtype=float)
assert a[0] == b[0]
