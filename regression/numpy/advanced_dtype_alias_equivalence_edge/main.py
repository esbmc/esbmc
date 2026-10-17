import numpy as np

a = np.array([-2.7], dtype=int)
b = np.array([-2.7], dtype=np.int64)
c = np.array([-2.7], dtype="int64")
d = np.array([-2.7], dtype=np.int_)
assert a[0] == -2
assert a[0] == b[0] and b[0] == c[0] and c[0] == d[0]
