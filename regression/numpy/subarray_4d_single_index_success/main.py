import numpy as np

a = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
v = a[1]

assert v[0][0][0] == 9
assert v[1][1][1] == 16
assert len(v) == 2
assert v.shape[0] == 2
assert v.shape[1] == 2
assert v.shape[2] == 2
assert v.ndim == 3
assert v.size == 8
