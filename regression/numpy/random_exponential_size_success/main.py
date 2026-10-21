import numpy as np

a = np.random.exponential(1.0, 3)

assert len(a) == 3
assert a[2] >= 0.0
