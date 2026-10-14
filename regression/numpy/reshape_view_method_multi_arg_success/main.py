import numpy as np

a = np.array([1, 2, 3, 4])
r = a.reshape(2, 2)

r[0][1] = 9

assert a[1] == 9
