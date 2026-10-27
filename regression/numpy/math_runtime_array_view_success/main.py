import numpy as np

a = np.array([0.5, 1.0, 2.0, 4.0])
v = a[1::2]
a[1] = 2.0
b = np.exp(v)
assert b[0] > 7.38 and b[1] > 54.59
