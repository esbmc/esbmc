import numpy as np

a = np.arange(-3, 3, 2.5, dtype=np.int64)
b = np.arange(0.5, 3.5, dtype=np.int32)
assert len(a) == 3 and a[1] == 0 and a[2] == 3
assert len(b) == 3 and b[0] == 0 and b[2] == 2
