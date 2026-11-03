import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
c = nondet_bool()
if c:
    v = a[0]
else:
    v = a[1]
assert len(v) == 3
assert v.shape == (3,)
if c:
    assert v[2] == 3
else:
    assert v[2] == 6
