import numpy as np

a = np.array([1, 2, 3])
for x in np.nditer(a, op_flags=["readwrite"]):
    x[...] = 2.7
assert a[0] == 2.7
