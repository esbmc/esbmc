import numpy as np

a = np.array([3.9], dtype=np.int_)
b = np.array([3.9], dtype="int64")
c = np.array([3.9], dtype=int)
d = np.array(["q"], dtype=np.str_)
assert a[0] == 3 and b[0] == 3 and c[0] == 3
assert d[0] == "q"
