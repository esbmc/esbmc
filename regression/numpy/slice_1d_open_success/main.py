import numpy as np

a = np.array([10, 20, 30, 40, 50])
b = a[2:]
assert b[0] == 30
assert b[1] == 40
assert b[2] == 50
