import numpy as np

a = np.random.normal(size=(0, 3))

assert a.shape == (0, 3)
assert a.size == 0
