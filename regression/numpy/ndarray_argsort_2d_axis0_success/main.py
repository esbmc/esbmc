import numpy as np

a = np.array([[3, 1], [2, 4]])
idx = a.argsort(axis=0)

assert idx[0, 0] == 1
assert idx[0, 1] == 0
