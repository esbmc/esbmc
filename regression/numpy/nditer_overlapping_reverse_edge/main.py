import numpy as np

a = np.array([1, 2, 3, 4])
it = np.nditer([a[1:4], a[0:3]], op_flags=[["readonly"], ["readwrite"]])
for x, y in it:
    y[...] = x
assert a[0] == 2
assert a[1] == 3
assert a[2] == 4
assert a[3] == 4
