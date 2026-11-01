import numpy as np

a = np.array([1.9, 2.5, -1.9])
b = a.astype(int)

assert b[0] == 1
assert b[1] == 2
assert b[2] == -1

n = np.array([300, -1])
w = n.astype(np.int8)
assert w[0] == 44 and w[1] == -1
assert w.dtype == np.int8 and w.itemsize == 1 and w.nbytes == 2
u = n.astype("u1")
assert u[1] == 255 and u.dtype == "uint8"
s = np.array([0.1]).astype("f4")
assert float(s[0]) != 0.1 and s.itemsize == 4
h = w.astype(np.int16)
h[0] += 1000
assert h[0] == 1044 and h.itemsize == 2
v = w[1:]
assert v[0] == -1 and v.nbytes == 1 and v.dtype == np.int8

a_m1 = np.array([True, False, True])
b_m1 = a_m1.astype(int)

assert b_m1[0] == 1
assert b_m1[1] == 0
assert b_m1[2] == 1

c_m1 = np.array([1, 2, 3])
d_m1 = c_m1.astype(np.int64)
assert d_m1[0] == 1 and d_m1[2] == 3
assert len(d_m1) == len(c_m1)
d_m1[0] = 9
assert c_m1[0] == 1
