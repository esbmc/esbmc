import numpy as np

a = np.linspace(1, 2, 3, dtype=np.int64)
v = a[1:]
assert v[0] == 1 and v[1] == 2
