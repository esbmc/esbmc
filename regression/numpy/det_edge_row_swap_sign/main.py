import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[3, 4], [1, 2]])
da = np.linalg.det(a)
db = np.linalg.det(b)
assert db == -da
