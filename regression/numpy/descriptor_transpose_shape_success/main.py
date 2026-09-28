import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
t = np.transpose(a)

assert len(t) == 3
assert t.shape[0] == 3
assert t.shape[1] == 2
assert t.ndim == 2
