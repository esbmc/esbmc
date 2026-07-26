import numpy as np

x = np.array([[1, 2], [3, 4]])
row = x[0]
x[0][0] = 7

assert row[0] == 1
