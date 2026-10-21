import numpy as np

a = np.random.normal(0.0, 1.0, size=3)

assert a.shape == (3,)
assert a[0] == a[0]
