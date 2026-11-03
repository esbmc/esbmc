import numpy as np

a = np.random.choice(a=[1, 2], size=2, replace=True)
b = np.random.choice([1, 2], 2, True)

assert a.shape == b.shape
assert a[1] == 1 or a[1] == 2
