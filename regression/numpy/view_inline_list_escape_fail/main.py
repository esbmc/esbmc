import numpy as np

x = np.array([[1, 2], [3, 4]])
holder = [x[0]]
holder[0][0] = 999

assert x[0][0] == 1
