import numpy as np

a = np.array([[1, 3, 5], [2, 4, 6]])
# NumPy rejects searchsorted over a full 2-D matrix.
idx = np.searchsorted(a, 4)

assert idx[0] == 1
