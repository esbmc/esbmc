import numpy as np

a = np.array([[1, 2], [3, 4]])

assert a.flat[0] == 1
assert a.flat[1] == 2
assert a.flat[2] == 3
assert a.flat[3] == 4
