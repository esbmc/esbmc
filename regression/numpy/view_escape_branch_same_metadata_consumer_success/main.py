import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
c = nondet_bool()
if c:
    v = a[0]
else:
    v = a[1]
total = np.sum(v)
values = v.tolist()
if c:
    assert total == 6
    assert values == [1, 2, 3]
else:
    assert total == 15
    assert values == [4, 5, 6]
