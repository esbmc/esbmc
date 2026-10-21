import numpy as np

a = np.random.binomial(4, 0.5, size=3)

assert a[1] >= 0
assert a[1] <= 4
