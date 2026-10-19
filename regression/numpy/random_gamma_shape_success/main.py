import numpy as np

a = np.random.gamma(2.0, size=(1, 2, 2))

assert a.shape == (1, 2, 2)
assert a[0][1][1] >= 0.0
