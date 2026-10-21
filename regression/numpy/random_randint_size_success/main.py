import numpy as np

a = np.random.randint(1, 4, size=(2, 2))

assert a.shape == (2, 2)
assert a[1][0] >= 1
assert a[1][0] < 4
