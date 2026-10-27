import numpy as np

a = np.array([1, 2, 3])
b = np.zeros(3, dtype=int)
c = np.ones(3, dtype=int)
it = np.nditer([a, b, c],
               op_flags=[["readonly"], ["writeonly"], ["readwrite"]])
for x, y, z in it:
    y[...] = x * 2
    z[...] = z + x
assert b[2] == 6
assert c[1] == 3
assert a[0] == 1
