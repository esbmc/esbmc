import numpy as np

a = np.array([[1, 2], [3, 4]])
row = a[0]
a = np.array([[9, 9], [9, 9]])
assert row[0] == 1
assert row[1] == 2
