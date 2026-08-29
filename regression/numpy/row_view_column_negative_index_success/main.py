import numpy as np

a = np.array([[1, 2], [3, 4]])
row = a[1]

assert row[-1] == 4
assert row[-2] == 3

