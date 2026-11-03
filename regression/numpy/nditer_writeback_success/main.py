import numpy as np

a = np.array([1, 2, 3])
with np.nditer(a, op_flags=["readwrite"]) as it:
    for x in it:
        x[...] = x + 100
assert a[0] == 101 and a[2] == 103

b = np.array([4, 5])
for x in np.nditer(b, op_flags=["readwrite"]):
    x[()] = 7
assert b[0] == 7 and b[1] == 7
