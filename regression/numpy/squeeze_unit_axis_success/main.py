import numpy as np

a = np.array([[[1, 2, 3]]])
b = np.squeeze(a)

assert b[0] == 1
assert b[2] == 3
