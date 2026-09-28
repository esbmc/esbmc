import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)

a.flat[0] = 9

assert t[0][0] == 9
