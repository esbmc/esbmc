import numpy as np

a = np.random.lognormal(size=(2, 2))

assert a.ndim == 2
assert a[1][1] >= 0.0
