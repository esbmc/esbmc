import numpy as np

axis = int(input())
a = np.array([[1, 2], [3, 4]])
t = np.moveaxis(a, axis, 1)

assert t[0][0] == 1
