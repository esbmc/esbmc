import numpy as np

a = np.random.beta(1.0, 2.0, (3,))

assert a.size == 3
assert a[0] >= 0.0
assert a[0] <= 1.0
