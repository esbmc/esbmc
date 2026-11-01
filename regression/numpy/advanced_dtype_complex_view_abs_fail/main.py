import numpy as np

a = np.array([3 + 4j])
b = a.reshape(1, 1)
assert np.abs(b)[0][0] == 5.0
