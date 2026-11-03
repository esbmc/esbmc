import numpy as np

a = np.array([1, 2, 3])
c = a.copy()
for x in np.nditer(c, op_flags=["readwrite"]):
    x[...] = 0
assert c[0] == 0 and c[2] == 0
assert a[0] == 1 and a[1] == 2 and a[2] == 3
