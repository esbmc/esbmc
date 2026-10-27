import numpy as np

a = np.random.choice([1, 2, 3, 4], (2, 2), replace=False)

assert a.shape == (2, 2)
assert a[0][0] != a[1][1]
assert a[0][1] != a[1][0]
