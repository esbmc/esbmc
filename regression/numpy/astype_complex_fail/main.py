import numpy as np

a = np.array([1, 2, 3])
b = a.astype(np.complex128)
assert b[1] == 2 + 1j
