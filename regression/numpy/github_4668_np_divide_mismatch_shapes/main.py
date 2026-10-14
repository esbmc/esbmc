import numpy as np

a = np.array([8, 6, 4])
b = np.array([2, 3])
c = np.divide(a, b)
assert c[0] == 4
