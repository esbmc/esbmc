import numpy as np

x = nondet_float()
if x >= 0.0 and x <= 1.0:
    a = np.array([[2.0, x, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]])
    e = np.linalg.eigvals(a)
    assert e.dtype == np.complex128
    assert abs((e[0] + e[1] + e[2]).real - 10.0) < 1e-6
    w, v = np.linalg.eig(a)
    assert w.dtype == np.complex128 and v.dtype == np.complex128
    r = a[1][0] * v[0][2] + a[1][1] * v[1][2] + a[1][2] * v[2][2] - w[2] * v[1][2]
    assert abs(r.real) < 1e-6 and abs(r.imag) < 1e-6
    n = v[0][2].real * v[0][2].real + v[0][2].imag * v[0][2].imag + v[1][2].real * v[1][2].real + v[1][2].imag * v[1][2].imag + v[2][2].real * v[2][2].real + v[2][2].imag * v[2][2].imag
    assert abs(n - 1.0) < 1e-6
