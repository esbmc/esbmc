import numpy as np

a = np.array([1 + 2j, 3j])
b = np.array([1j, 2])
c = a * 2
assert c[0] == 2 + 4j and c[1] == 6j
d = a + b
assert d[0] == 1 + 3j and d[1] == 2 + 3j
e = a - b
assert e[1] == -2 + 3j
f = a * b
assert f[0] == -2 + 1j
g = a / b
assert g[0] == 2 - 1j and g[1] == 1.5j
h = -a
assert h[0] == -1 - 2j
i = a + 1.5
assert i[1] == 1.5 + 3j
j = 1j * a
assert j[1] == -3 + 0j
k = np.add(a, b)
assert k[1] == 2 + 3j
l = np.multiply(a, 2)
assert l[0] == 2 + 4j
n = np.array([1 + 1j, 1j]) / np.array([0, 1])
assert n[1] == 1j
m = np.array([[1j, 2], [3, 4j]])
o = m + np.array([1, 1j])
assert o[1][1] == 5j and o.nbytes == 64

s = np.array([0.1 + 0.2j], dtype=np.complex64)
assert s.itemsize == 8 and s.nbytes == 8 and s.dtype == np.complex64
t = s * 2
assert t.itemsize == 8 and t.dtype == "c8"
u = s + np.array([1j])
assert u.itemsize == 16
assert float(s[0].real) != 0.1
widened = np.array([1, 2, 3]).astype(np.complex128)
assert widened[1] == 2 + 0j
assert widened.dtype == np.complex128 and widened.itemsize == 16 and widened.nbytes == 48
from_text = np.array([1.5]).astype("c16")
assert from_text[0].real == 1.5 and from_text[0].imag == 0.0
