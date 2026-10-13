import numpy as np
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])
flag = True
if flag:
    v = a[0:]
else:
    v = b[1:]
v[0] = 7
assert b[1] == 7
