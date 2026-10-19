import numpy as np

a = np.full(2, 1 - 1j)
b = np.full(2, 3, dtype=complex)
assert a[1] == 1 - 1j
assert b[0] == 3 and b[0].imag == 0.0
