import numpy as np
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])
flag = False
if flag:
    v = a[0:]
else:
    v = b[1:]
w = v[1:]
w[0] = 0
assert b[2] == 0
c = v.copy()
v[0] = 5
assert c[0] == 20
assert len(w) == 2
