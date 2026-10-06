import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
c = nondet_bool()
v = a[0]
if c:
    v = a[1]
v[2] = 9
if c:
    assert a[1][2] == 9
    assert a[0][2] == 3
else:
    assert a[0][2] == 9
    assert a[1][2] == 6
