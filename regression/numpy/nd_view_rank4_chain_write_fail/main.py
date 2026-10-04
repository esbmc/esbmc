import numpy as np

a = np.zeros((2, 2, 2, 2))
v = a[1]
w = v[0]
w[1][1] = 5
assert a[1][0][1][1] == 0
assert w.shape == (2, 2)
