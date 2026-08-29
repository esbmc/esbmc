import numpy as np

x = np.array([[1, 2], [3, 4]])
holder = [x[0][0]]

assert holder[0] == 1
assert x[0][0] == 1
