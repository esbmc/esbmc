import numpy as np

a = np.random.chisquare(2, size=(3, 1))

assert a.shape == (3, 1)
assert a[2][0] >= 0.0
