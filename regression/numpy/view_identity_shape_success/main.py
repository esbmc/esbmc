import numpy as np

a = np.array([[1, 2], [3, 4]])
row = a[0]
shape = row.shape

assert shape[0] == 2
