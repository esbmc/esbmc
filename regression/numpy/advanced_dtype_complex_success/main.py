import numpy as np

a = np.array([1 + 2j, 3j])
b = np.ones(2, dtype=complex)
assert a[0] == 1 + 2j
assert a[1].imag == 3.0
assert b[1] == 1
