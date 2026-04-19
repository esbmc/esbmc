import numpy as np
a = np.transpose([[1, 2], [3, 4]])
assert a[0][0] == 1
assert a[0][1] == 3
assert a[1][0] == 2
assert a[1][1] == 4
