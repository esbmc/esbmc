import numpy as np
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])
flag = True
if flag:
    v = a[0:]
else:
    v = b[1:]
b = np.array([5, 6])
assert v[0] == 1
