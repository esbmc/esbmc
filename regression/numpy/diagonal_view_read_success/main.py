import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
d = np.diagonal(a)

assert len(d) == 3
assert d[0] == 1
assert d[1] == 5
assert d[2] == 9
