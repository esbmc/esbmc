import numpy as np

a = np.array([7, -3], dtype="int16")
b = np.ones(3, dtype="uint32")
assert a[0] == 7 and a[1] == -3
assert b[2] == 1
