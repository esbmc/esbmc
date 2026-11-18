import numpy as np

a = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
w = np.linalg.eigvals(a)
assert w.dtype == np.complex128
assert abs(w[0].real + w[1].real + w[2].real - 6.0) < 1e-9
assert abs(w[0].imag) < 1e-9 and abs(w[1].imag) < 1e-9 and abs(w[2].imag) < 1e-9
assert abs(w[0].real - 1.0) < 1e-9 or abs(w[0].real - 4.0) < 1e-9
