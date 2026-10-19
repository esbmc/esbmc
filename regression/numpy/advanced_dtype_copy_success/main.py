import numpy as np

a = np.full(3, 2.5, dtype=np.int64)
c = a.copy()
a[0] = 9
assert c[0] == 2 and a[0] == 9
