import numpy as np

a = np.array([0, 1, 2, 3, 4, 5])
v = a[::2]
w = np.array([1, 2, 3])
total = 0
for x, y in np.nditer([v, w]):
    assert x == 2 * (y - 1)
    total += x * y
assert total == 16
