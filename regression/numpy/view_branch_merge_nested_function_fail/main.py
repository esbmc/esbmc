import numpy as np
a = np.array([1, 2, 3, 4, 5])
v = a[0:2]
def f():
    return len(v)
flag = True
if flag:
    v = a[1:5]
assert f() == 4
