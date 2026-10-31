import numpy as np

a = np.array([1, 2, 3])
b = a.astype(np.complex128)
assert b[1] == 2 + 0j
assert b.dtype == np.complex128 and b.itemsize == 16 and b.nbytes == 48
c = np.array([1.5]).astype("c16")
assert c[0].real == 1.5 and c[0].imag == 0.0
