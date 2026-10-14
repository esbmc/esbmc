import numpy as np

c = np.arange(24).reshape(2, 3, 4)
t = np.transpose(c)
assert t[3][2][1] == 23
u = np.swapaxes(c, 0, 2)
assert u[1][1][1] == 16
m = np.moveaxis(c, 0, -1)
assert m[2][3][1] == 23
u[0][0][1] = 99
assert c[1][0][0] == 99
