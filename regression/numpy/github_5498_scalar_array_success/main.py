# A scalar % ndarray broadcasts the scalar over the array (#5498).
import numpy as np

a = np.array([4, 5, 6])
b = 10 % a
assert b[0] == 2
