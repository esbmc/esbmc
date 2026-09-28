import numpy as np

a = np.array([2, 1, 3, 1, 2])

assert a.argmin() == 1
assert a.argmax() == 2
