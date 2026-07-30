import numpy as np

a = np.array([[3, 1], [4, 2]])
b = np.sort(a, axis=1)

assert b[0][0] == 1
