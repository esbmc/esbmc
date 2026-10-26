import numpy as np

a = np.arange(6, dtype=np.int64)
v = a[::-2]
assert len(v) == 3
assert v[0] == 5 and v[2] == 1
