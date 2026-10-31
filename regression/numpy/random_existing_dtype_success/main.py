import numpy as np

a = np.random.randint(-3, 3, size=3, dtype=np.int16)

assert a.shape == (3,)
assert a[0] >= -3
assert a[2] < 3

b = np.random.randint(0, 100, size=2, dtype=np.int8)
assert b.itemsize == 1
b[0] = b[0] + 100
assert b[0] != 200
