import numpy as np

a = np.array([2, 3])
b = np.multiply(False, a)
assert b[0] == 0
