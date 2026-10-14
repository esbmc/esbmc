import numpy as np

a = np.array([[1, 2], [3, 4]])
# NumPy rejects searchsorted over a full 2-D matrix.
a.searchsorted(2)
