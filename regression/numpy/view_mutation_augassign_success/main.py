import numpy as np

a = np.array([[1, 2], [3, 4]])
row = a[0]

row[0] += 1

assert row[0] == 2
assert a[0][0] == 2
