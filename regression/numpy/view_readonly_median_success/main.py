import numpy as np

a = np.array([[1, 5, 3], [2, 4, 6]])
row = a[0]
m = np.median(row)

assert m == 3
