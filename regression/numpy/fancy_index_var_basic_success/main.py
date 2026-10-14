import numpy as np

a = np.array([10, 20, 30, 40])
idx = [0, 2]
b = a[idx]
assert b[0] == 10
assert b[1] == 30
