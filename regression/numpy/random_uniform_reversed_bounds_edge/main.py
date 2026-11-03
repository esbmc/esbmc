import numpy as np

a = np.random.uniform(2.0, -2.0, size=2)

assert a[0] >= -2.0
assert a[0] <= 2.0
