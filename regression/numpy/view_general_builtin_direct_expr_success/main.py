import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
assert sum(a[1:4]) == 6
assert sorted(a[::2]) == [3, 4, 5]
assert list(a[1:3]) == [1, 4]
assert max(a[3:]) == 5
