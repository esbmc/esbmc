import numpy as np

a = np.multiply([[1], [2]], [3, 4])
assert a[0][0] == 3
assert a[0][1] == 4
assert a[1][0] == 6
assert a[1][1] == 8
