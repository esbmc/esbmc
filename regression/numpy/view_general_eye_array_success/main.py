import numpy as np

def make():
    return np.eye(3)

a = make()
assert a.shape == (3, 3)
assert a[1][1] == 1
assert a[1][2] == 0
e = np.eye(2, 3)
assert e.shape == (2, 3)
k = np.eye(3, k=1)
assert k[0][1] == 1
assert k[0][0] == 0
