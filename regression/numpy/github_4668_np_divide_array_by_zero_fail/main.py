import numpy as np

a = np.array([8, 6])
b = np.array([0, 3])
c = np.divide(a, b)
assert c[1] == 2
