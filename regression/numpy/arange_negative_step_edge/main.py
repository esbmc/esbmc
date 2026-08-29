import numpy as np

a = np.arange(5, 1, -2)
assert len(a) == 2
assert a[0] == 5
assert a[1] == 3

b = np.arange(5, 1, 2)
assert len(b) == 0
