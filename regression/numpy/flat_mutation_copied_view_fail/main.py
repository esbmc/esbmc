import numpy as np

a = np.array([[1, 2], [3, 4]])
t = a.T
a.flat[0] = 10
