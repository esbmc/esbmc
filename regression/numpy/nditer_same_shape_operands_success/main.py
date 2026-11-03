import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
t = np.transpose(a)
c = np.array([[1, 2], [3, 4], [5, 6]])
order = 0
for x, y in np.nditer([t, c]):
    order = order * 10 + x
assert order == 142536

order = 0
for x, y in np.nditer([t, t]):
    assert x == y
    order = order * 10 + x
assert order == 123456
