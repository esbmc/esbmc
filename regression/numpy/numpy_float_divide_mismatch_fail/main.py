import numpy as np

a = np.array([[8.0, 6.0], [9.0, 12.0]])
b = np.array([[2.0, 3.0], [3.0, 4.0]])
result = np.divide(a, b)
assert result[0][0] == 5.0
