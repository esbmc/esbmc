import numpy as np

a = np.full((2, 2), 5).flatten()

assert a[0] == 5
assert a[1] == 5
assert a[2] == 5
assert a[3] == 5
