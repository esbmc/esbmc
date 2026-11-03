import numpy as np

a = np.arange(1.0, 1.3, 0.1)
b = np.arange(0.0, 1.0, 0.25)
assert len(a) == 4
assert a[2] != 1.2
assert len(b) == 4 and b[3] == 0.75
