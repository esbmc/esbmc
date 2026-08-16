import numpy as np

a = np.arange(2, 6)
assert len(a) == 4
assert a[0] == 2
assert a[1] == 3
assert a[2] == 4
assert a[3] == 5

b = np.arange(2, 8, 2)
assert len(b) == 3
assert b[0] == 2
assert b[1] == 4
assert b[2] == 6
