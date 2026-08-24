import numpy as np

x = np.array([[1, 2], [3, 4]])
b = x.ravel()
holder = [b]
holder[0][0] = 999

assert x[0][0] == 1
