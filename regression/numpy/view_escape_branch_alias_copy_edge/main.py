import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
c = nondet_bool()
if c:
    v = a[0]
else:
    v = a[1]
k = np.copy(v)
k[0] = 99
assert a[0][0] == 1
assert a[1][0] == 4
if c:
    assert k[1] == 2
else:
    assert k[1] == 5
