import numpy as np

a = np.array([3, 1, 2])
b = np.array([[3, 1], [4, 0]])

assert np.argmin(a) == 1
assert np.argmax(a) == 0
assert np.argmin(b) == 3
assert np.argmax(b) == 2
