import numpy as np

a = np.random.poisson(2.0, size=(2, 3))

assert a.shape == (2, 3)
assert a[1][2] >= 0
