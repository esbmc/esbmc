import numpy as np

a = np.random.randn(2, 2)

assert a.shape == (2, 2)
assert a[0][1] == a[0][1]
