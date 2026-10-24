import numpy as np

m = np.array([[1, 2], [3, 4]])
row = m[0]
for x in np.nditer(row, op_flags=["readwrite"]):
    x[...] = -x
assert m[0, 0] == -1 and m[0, 1] == -2
assert m[1, 0] == 3 and m[1, 1] == 4

a = np.array([0, 1, 2, 3, 4, 5])
for x in np.nditer(a[::2], op_flags=["readwrite"]):
    x[...] = 9
assert a[0] == 9 and a[1] == 1 and a[2] == 9 and a[4] == 9 and a[5] == 5
