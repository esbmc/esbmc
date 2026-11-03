import numpy as np

a = np.full(2, 9.9, dtype="int32")
b = np.full(2, 9.9, dtype=np.int32)
assert a[0] == b[0] and b[1] == 9
