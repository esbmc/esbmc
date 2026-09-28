import numpy as np

a = np.array([[1, 2], [3, 4]])
row = a[0]
a = a
row[0] = 9

assert a[0][0] == 9
