import numpy as np

a = np.array([[3, 1], [2, 4]])
a.sort(axis=0)  # In-place sort by column

assert a[0, 0] == 2
assert a[0, 1] == 1
assert a[1, 0] == 3
assert a[1, 1] == 4
