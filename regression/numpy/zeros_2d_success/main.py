import numpy as np

a = np.zeros((2, 3))
assert a[0][0] == 0.0
assert a[1][2] == 0.0

a_alt = np.zeros((2, 1))
assert a_alt[0][0] == 0.0
assert a_alt[1][0] == 0.0
