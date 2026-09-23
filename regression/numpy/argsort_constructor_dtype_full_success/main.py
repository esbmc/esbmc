import numpy as np

idx = np.argsort(np.zeros((3,), dtype=int))

assert idx[0] == 0
assert idx[1] == 1
assert idx[2] == 2
