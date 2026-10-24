import numpy as np


def nothing(a):
    return a[2:2]


a = np.array([10, 20, 30, 40, 50])
e = nothing(a)
assert e.tolist() == []
assert np.sum(e) == 0
assert not e.any()
assert e.all()
c = np.copy(e)
assert len(c) == 0
