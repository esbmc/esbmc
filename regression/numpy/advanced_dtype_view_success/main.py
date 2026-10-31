import numpy as np

a = np.array([1.9, 2.9, 3.9], dtype=np.int32)
v = a[1:]
assert v[0] == 2
a[2] = 5
assert v[1] == 5

b = np.array([100, -100, 50, 7], dtype=np.int8)
s = b[1:3]
s[0] = s[0] + s[0]
assert b[1] == 56 and s.itemsize == 1 and s.nbytes == 2
r = b[::-1]
r[0] += 121
assert b[3] == -128
m = np.array([[100, 27], [1, 2]], dtype=np.int8)
f = m.reshape(4)
f[0] += 100
assert m[0][0] == -56
t = m.T
t[0][1] = t[0][1] + 127
assert m[1][0] == -128
flat = b.ravel()
flat[0] += 100
assert b[0] == -56 and flat.itemsize == 1
first = b[1]
twice = first + first
assert twice == 112
minus = -m
assert minus[0][1] == -27 and minus.itemsize == 1
