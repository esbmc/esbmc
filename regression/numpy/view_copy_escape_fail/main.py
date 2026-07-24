import numpy as np

x = np.array([[1, 2], [3, 4]])
row = x[0]
holder = [row]
x[0][0] = 5

assert holder[0][0] == 1
