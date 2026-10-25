import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([1, 2, 3])
c = a + b
assert c[0, 0] == 2
