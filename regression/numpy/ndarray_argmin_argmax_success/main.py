import numpy as np

a = np.array([3, 1, 4, 1, 5])

assert a.argmin() == 1
assert a.argmax() == 4
