import numpy as np

c = np.array([[3, 1, 2], [6, 4, 5]])

amn0 = np.argmin(c, axis=0)
amx1 = np.argmax(c, axis=1)

assert amn0[0] == 0
assert amn0[1] == 0
assert amn0[2] == 0
assert amx1[0] == 0
assert amx1[1] == 0
