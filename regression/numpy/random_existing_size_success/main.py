import numpy as np

a = np.random.random(size=3)

assert a.shape == (3,)
assert a[2] >= 0.0
assert a[2] < 1.0
