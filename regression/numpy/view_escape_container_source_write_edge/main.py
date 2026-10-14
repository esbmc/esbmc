import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
box = {'low': a[1]}
a[1][2] = 7
assert box['low'][2] == 7
