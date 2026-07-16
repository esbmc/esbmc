import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = a[1::2]

assert b[0] == 2
assert b[1] == 4
