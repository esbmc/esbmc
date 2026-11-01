import numpy as np

a = np.array([3, 1, 4, 1, 5])

assert a.argmin() == 1
assert a.argmax() == 4

a_alt = np.array([2, 1, 3, 1, 2])

assert a_alt.argmin() == 1
assert a_alt.argmax() == 2
