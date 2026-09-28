import numpy as np

a = np.sort(np.full((3,), 5.7, dtype=int))

assert a[0] == 5
assert a[1] == 5
assert a[2] == 5
