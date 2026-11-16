import numpy as np

a = np.array([3 + 4j, 1j])
b = np.abs(a)
assert b[0] == 5.0
