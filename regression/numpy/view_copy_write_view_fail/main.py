import numpy as np

x = np.array([[1, 2], [3, 4]])
row = x[0]
row[1] = 8

assert x[0][1] == 2
