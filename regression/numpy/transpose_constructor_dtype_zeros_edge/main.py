import numpy as np

a = np.zeros((2, 3), dtype=int).T

assert a.shape[0] == 3
assert a.shape[1] == 2
assert a[0][0] == 0
