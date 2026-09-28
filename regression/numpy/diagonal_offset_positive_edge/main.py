import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
d = np.diagonal(a, offset=1)

assert len(d) == 2
assert d[0] == 2
assert d[1] == 6
