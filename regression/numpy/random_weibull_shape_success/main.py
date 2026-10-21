import numpy as np

a = np.random.weibull(2.0, size=(2,))

assert a.shape == (2,)
assert a[1] >= 0.0
