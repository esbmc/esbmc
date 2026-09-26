import numpy as np

d = np.array([[1, 2], [3, 4]]).diagonal()
assert d[0] == 1
assert d[1] == 4
