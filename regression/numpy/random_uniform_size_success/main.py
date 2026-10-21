import numpy as np

a = np.random.uniform(-2.0, 2.0, size=4)

assert len(a) == 4
assert a[3] >= -2.0
assert a[3] <= 2.0
