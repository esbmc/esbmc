import numpy as np

a = np.full((2, 3), 4, dtype=np.int16)
assert a[1][2] == 4
assert len(a) == 2 and len(a[0]) == 3
