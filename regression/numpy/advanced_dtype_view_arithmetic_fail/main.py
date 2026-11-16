import numpy as np

a = np.array([1, 5])
v = a[0:1]
b = v + 1
assert b[0] == 5
