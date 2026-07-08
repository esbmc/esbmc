import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
mask = np.array([True, False, False])
mask = np.array([False, False, True])
b = a[mask]
