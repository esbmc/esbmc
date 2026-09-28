import numpy as np

a = np.array([[[1, 2, 3]]])
v = np.squeeze(a)

assert v[0] == 1
assert v[2] == 3
