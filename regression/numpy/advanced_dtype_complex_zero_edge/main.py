import numpy as np

a = np.zeros(2, dtype=np.complex64)
b = np.array([0j, 2])
assert a[0] == 0j
assert a[1].imag == 0.0
assert b[1] == 2 and b[1].imag == 0.0
