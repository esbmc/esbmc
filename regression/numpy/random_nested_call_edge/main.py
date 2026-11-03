import numpy as np

x = np.random.normal(np.random.uniform(0.0, 1.0), 0.0)
a = np.random.poisson(np.random.uniform(0.0, 5.0), size=2)

assert x >= 0.0
assert x <= 1.0
assert a[1] >= 0
