import numpy as np
a = np.array([True, False, True])
b = np.array([False, True, True])
result = np.dot(a, b)
assert result == 1  # True*False + False*True + True*True = 1
