import numpy as np

a = np.array([[8, 6], [4, 2]])
b = np.array([[1, 1], [1, 1]])
c = np.subtract(a, b)
assert c[1][1] == 1
