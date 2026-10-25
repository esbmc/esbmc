import numpy as np

a = np.array([1, 2])
b = a
assert b[1] != 2
for i in range(2):
    a = np.array([i, i])
