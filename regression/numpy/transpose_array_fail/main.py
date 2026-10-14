import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.transpose(a)

assert b[0][0] == 2
