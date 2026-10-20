import numpy as np

a = np.array([1, 2, 3, 4])
x = a[2:2]
y = a[2:2]
a = np.array([9, 9, 9, 9])
assert len(y) == 0
assert y.tolist() == []
assert x.tolist() == []
assert np.sum(y) == 0
assert y.size == 0
