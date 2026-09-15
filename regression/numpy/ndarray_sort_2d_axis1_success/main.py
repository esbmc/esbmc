import numpy as np

a = np.array([[3, 1], [4, 2]])
a.sort(axis=1)  # In-place sort by row

assert a[0, 0] == 1
assert a[1, 0] == 2
