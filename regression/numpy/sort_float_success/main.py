import numpy as np

a = np.array([2.5, 1.25, 3.0])
b = np.sort(a)

assert b[0] == 1.25
assert b[1] == 2.5
assert b[2] == 3.0
