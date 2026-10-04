import numpy as np

a = np.array([[[1, 2], [3, 4]]])
t = a.transpose((0, 2, 1))

assert t.shape == (1, 2, 2)
assert t[0][1][0] == 2
