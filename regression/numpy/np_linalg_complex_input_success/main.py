import numpy as np

a = np.array([[1.0 + 1.0j, 2.0, 0.0], [0.0, 2.0 + 0.5j, 1.0], [1.0j, 0.0, 3.0]])
w, v = np.linalg.eig(a)
assert w.dtype == np.complex128 and v.dtype == np.complex128
t = w[0] + w[1] + w[2]
assert abs(t.real - 6.0) < 1e-9 and abs(t.imag - 1.5) < 1e-9
r = a[0][0] * v[0][0] + a[0][1] * v[1][0] + a[0][2] * v[2][0] - w[0] * v[0][0]
assert abs(r.real) < 1e-6 and abs(r.imag) < 1e-6
n = (
    v[0][0].real * v[0][0].real + v[0][0].imag * v[0][0].imag
    + v[1][0].real * v[1][0].real + v[1][0].imag * v[1][0].imag
    + v[2][0].real * v[2][0].real + v[2][0].imag * v[2][0].imag
)
assert abs(n - 1.0) < 1e-6
assert a[2][0] == 1.0j
