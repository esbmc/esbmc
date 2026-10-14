import numpy as np

a = np.add([10, 20, 30], [[1, 2, 3], [4, 5, 6]])
assert a[0][0] == 11
assert a[0][1] == 22
assert a[1][0] == 14
assert a[1][2] == 36
