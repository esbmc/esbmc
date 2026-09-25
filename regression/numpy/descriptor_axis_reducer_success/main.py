import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)

s0 = np.sum(t, axis=0)

assert s0[0] == 3
assert s0[1] == 7
