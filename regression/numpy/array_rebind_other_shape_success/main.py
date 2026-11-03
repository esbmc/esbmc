import numpy as np

a = np.array([1, 2])
assert a[1] == 2
first = a[0] + a[1]
a = np.array([5.5, 6.5, 7.5])
assert a[2] == 7.5
assert len(a) == 3
assert a.sum() == 19.5
assert first == 3
