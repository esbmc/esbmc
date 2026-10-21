import numpy as np

a = np.zeros(2)
a[0] = np.random.random()
a[1] = np.random.uniform(2.0, 3.0)

assert a[0] < 1.0
assert a[1] >= 2.0
