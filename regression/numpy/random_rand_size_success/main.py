import numpy as np

a = np.random.rand(2, 3)

assert a.shape == (2, 3)
assert a[1][2] >= 0.0
assert a[1][2] < 1.0
