import numpy as np

a = np.array([3, 1, 2])
result = a.sort()

assert result is None
assert a[0] == 1
