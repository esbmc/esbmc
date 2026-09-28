import numpy as np

a = np.arange(0.0, 1.0, 0.5)
assert len(a) == 2
assert a[0] == 0.0
assert a[1] == 0.5

b = np.arange(1, 3.5, 1)
assert len(b) == 3
assert b[0] == 1.0
assert b[1] == 2.0
assert b[2] == 3.0
