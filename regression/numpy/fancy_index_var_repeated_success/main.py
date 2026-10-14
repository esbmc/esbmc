import numpy as np

a = np.array([10, 20, 30, 40])
idx = [2, 0, 2]
b = a[idx]
assert b[0] == 30
assert b[1] == 10
assert b[2] == 30
