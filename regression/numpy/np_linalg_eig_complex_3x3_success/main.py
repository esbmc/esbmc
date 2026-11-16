import numpy as np

a = np.array([[1.0, 2.0, 1.0], [-2.0, 1.0, 0.0], [1.0, 0.0, 3.0]])
w, v = np.linalg.eig(a)
assert w.dtype == np.complex128 and v.dtype == np.complex128
assert abs((w[0] + w[1] + w[2]).real - 5.0) < 1e-6
assert w[1] == w[0].conjugate() or w[2] == w[1].conjugate()
r0 = a[0][0] * v[0][0] + a[0][1] * v[1][0] + a[0][2] * v[2][0] - w[0] * v[0][0]
r1 = a[1][0] * v[0][0] + a[1][1] * v[1][0] + a[1][2] * v[2][0] - w[0] * v[1][0]
r2 = a[2][0] * v[0][0] + a[2][1] * v[1][0] + a[2][2] * v[2][0] - w[0] * v[2][0]
assert abs(r0) < 1e-6 and abs(r1) < 1e-6 and abs(r2) < 1e-6
assert abs(v[0][0].real * v[0][0].real + v[0][0].imag * v[0][0].imag + v[1][0].real * v[1][0].real + v[1][0].imag * v[1][0].imag + v[2][0].real * v[2][0].real + v[2][0].imag * v[2][0].imag - 1.0) < 1e-6

e = np.linalg.eigvals(a)
assert e[1] == e[0].conjugate() or e[2] == e[1].conjugate()
assert abs(e[0].imag) > 1.8 or abs(e[1].imag) > 1.8
