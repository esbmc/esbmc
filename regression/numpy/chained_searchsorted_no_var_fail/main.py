import numpy as np

a = np.array([1, 2, 3]).searchsorted(2)

assert a == 0
