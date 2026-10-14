import numpy as np

a = np.add([[1, 2, 3], [4, 5, 6]], [10, 20, 30])
assert a[0][0] == 11
assert a[0][2] == 33
assert a[1][1] == 25
assert a[1][2] == 36
