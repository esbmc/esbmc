import numpy as np

a = np.full(3, 4, dtype=np.int16)
v = a[3:]
assert len(v) == 0
assert len(a) == 3
