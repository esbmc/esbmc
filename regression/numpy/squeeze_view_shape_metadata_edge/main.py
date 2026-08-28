import numpy as np

a = np.array([[1, 2, 3]])
v = np.squeeze(a, 0)

assert len(v) == 3
assert v.ndim == 1
assert v.shape[0] == 3
