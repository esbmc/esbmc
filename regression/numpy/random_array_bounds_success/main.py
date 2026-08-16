import numpy as np

a = np.random.random(2)

assert len(a) == 2
assert a[0] >= 0.0
assert a[0] < 1.0
assert a[1] >= 0.0
assert a[1] < 1.0
