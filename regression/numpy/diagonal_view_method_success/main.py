import numpy as np

a = np.array([[1, 2], [3, 4]])
d = a.diagonal()

assert len(d) == 2
assert d[0] == 1
assert d[1] == 4
