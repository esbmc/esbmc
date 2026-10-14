import numpy as np

a = np.array([[8, 6], [4, 2]])
b = np.array([[2, 3], [2, 1]])
c = np.divide(a, b)
assert c[1][0] == 2
