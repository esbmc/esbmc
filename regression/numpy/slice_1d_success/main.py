import numpy as np

a = np.array([10, 20, 30, 40, 50])
b = a[1:4]
assert b[0] == 20
assert b[1] == 30
assert b[2] == 40
