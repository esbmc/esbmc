import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
t = np.transpose(a)

assert len(t) == 4
assert t.shape[0] == 4
assert t.shape[1] == 2
assert t[3][1] == 8
