import numpy as np

a = np.zeros((16, 16))
t = a.T
t[3][5] = 7
assert a[5][3] == 7

s = a[2:14]
assert len(s) == 12
