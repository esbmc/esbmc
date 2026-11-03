import numpy as np

a = np.random.choice([1, 2, 3], size=3, replace=False)

assert a[0] != a[1]
assert a[1] != a[2]
assert a[0] != a[2]
assert a[0] + a[1] + a[2] == 6
