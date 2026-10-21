import numpy as np

a = np.random.poisson(1.0, size=(2, 0))

assert a.shape == (2, 0)
assert len(a) == 2
