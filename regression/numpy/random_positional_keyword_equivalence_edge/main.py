import numpy as np

a = np.random.uniform(1.0, 1.0, 2)
b = np.random.uniform(high=1.0, low=1.0, size=2)

assert a[0] == 1.0
assert b[1] == 1.0
