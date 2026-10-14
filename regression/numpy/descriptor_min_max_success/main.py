import numpy as np

a = np.array([[3, 1], [4, 2]])
t = np.transpose(a)

assert np.min(t) == 1
assert np.max(t) == 4
