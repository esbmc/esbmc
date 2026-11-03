import numpy as np

a = np.array([1, 2, 3, 4])
it = np.nditer([a[0:3], a[1:4]], op_flags=[["readonly"], ["readwrite"]])
for x, y in it:
    y[...] = x + 10
assert a[1] == 11
assert a[2] == 21
assert a[3] == 31
