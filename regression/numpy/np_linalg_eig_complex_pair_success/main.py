import numpy as np

a = np.array([[1.0, -2.0], [1.0, 3.0]])
w, v = np.linalg.eig(a)
assert w.dtype == np.complex128 and v.dtype == np.complex128
assert abs(w[0].real - 2.0) < 1e-9 and abs(w[0].imag - 1.0) < 1e-9
assert w[1] == w[0].conjugate()
assert v[0][1] == v[0][0].conjugate() and v[1][1] == v[1][0].conjugate()
r0 = a[0][0] * v[0][0] + a[0][1] * v[1][0] - w[0] * v[0][0]
r1 = a[1][0] * v[0][0] + a[1][1] * v[1][0] - w[0] * v[1][0]
assert abs(r0) < 1e-6 and abs(r1) < 1e-6
assert abs(v[0][0].real * v[0][0].real + v[0][0].imag * v[0][0].imag + v[1][0].real * v[1][0].real + v[1][0].imag * v[1][0].imag - 1.0) < 1e-9

e = np.linalg.eigvals(a)
assert e[1] == e[0].conjugate()

d, q = np.linalg.eig(np.array([[2.0, 1.0], [1.0, 3.0]]))
assert d.dtype == np.complex128 and q.dtype == np.complex128
assert d[0].imag == 0.0 and q[0][0].imag == 0.0
