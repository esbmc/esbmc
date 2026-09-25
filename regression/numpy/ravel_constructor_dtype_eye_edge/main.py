import numpy as np

a = np.ravel(np.full((2, 2), 3.5, dtype=int))

assert a[0] == 3
assert a[1] == 3
assert a[2] == 3
assert a[3] == 3
