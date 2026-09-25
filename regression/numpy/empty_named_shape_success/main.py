import numpy as np

shape = (2, 1)
a = np.empty(shape, dtype=int)
a[0][0] = 3
a[1][0] = 5

assert a[0][0] == 3
assert a[1][0] == 5
