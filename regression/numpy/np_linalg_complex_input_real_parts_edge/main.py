import numpy as np

a = np.array([[2.0 + 0.0j, 1.0], [0.0, 3.0 + 0.0j]])
w, v = np.linalg.eig(a)
assert w.dtype == np.complex128
assert abs(w[0].imag) < 1e-9 and abs(w[0].real + w[1].real - 5.0) < 1e-9
r = a[0][0] * v[0][1] + a[0][1] * v[1][1] - w[1] * v[0][1]
assert abs(r.real) < 1e-9 and abs(r.imag) < 1e-9
