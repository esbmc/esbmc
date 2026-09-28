import numpy as np

idx = np.argsort(np.array([30, 10, 20]))

assert idx[0] == 1
assert idx[1] == 2
assert idx[2] == 0
