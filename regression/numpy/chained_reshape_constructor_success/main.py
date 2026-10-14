import numpy as np

a = np.full((2, 2), 3).reshape((4,))

assert a[0] == 3
assert a[3] == 3
