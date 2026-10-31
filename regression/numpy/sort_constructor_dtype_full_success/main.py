import numpy as np

a = np.sort(np.full((3,), 5.7, dtype=int))

assert a[0] == 5
assert a[1] == 5
assert a[2] == 5

b = np.sort(np.array([100, -100], dtype=np.int8))
assert b[0] == -100 and b.itemsize == 1
b[1] += 100
assert b[1] == -56
