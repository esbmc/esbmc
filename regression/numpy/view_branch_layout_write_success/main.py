import numpy as np
a = np.array([1, 2, 3, 4, 5])
flag = False
if flag:
    v = a[0:2]
else:
    v = a[1:5]
assert len(v) == 4
assert v[3] == 5
v[0] = 9
assert a[1] == 9
