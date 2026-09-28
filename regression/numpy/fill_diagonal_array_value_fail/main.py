import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([10, 20, 30])
np.fill_diagonal(a, b)
