import numpy as np

a = np.random.gamma(1.0, 2.0, (2,))
b = np.random.gamma(shape=1.0, scale=2.0, size=(2,))

assert a.shape == b.shape
assert b[1] >= 0.0
