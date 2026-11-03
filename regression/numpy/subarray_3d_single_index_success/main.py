import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
row = a[0]

assert row[0][0] == 1
assert row[0][1] == 2
assert row[1][0] == 3
assert len(row) == 2
assert row.shape[0] == 2
assert row.shape[1] == 2
assert row.ndim == 2
assert row.size == 4
