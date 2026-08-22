import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
d = np.diagonal(a)

assert len(d) == 2
assert d[0] == 1
assert d[1] == 5
