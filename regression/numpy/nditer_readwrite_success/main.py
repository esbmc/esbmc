import numpy as np

a = np.array([1, 2, 3])
for x in np.nditer(a, op_flags=["readwrite"]):
    x[...] = 2 * x
    assert x % 2 == 0
assert a[0] == 2 and a[1] == 4 and a[2] == 6

m = np.array([[1, 2], [3, 4]])
for x in np.nditer(np.transpose(m), op_flags=["readwrite"]):
    x[...] = x + 10
assert m[0, 0] == 11 and m[0, 1] == 12 and m[1, 0] == 13 and m[1, 1] == 14
