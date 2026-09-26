import numpy as np

a = np.array([[1, 2], [3, 4]]).diagonal()

assert a[0] == 1
assert a[1] == 4
