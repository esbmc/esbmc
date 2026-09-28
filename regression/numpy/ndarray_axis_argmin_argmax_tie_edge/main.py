import numpy as np

c = np.array([[1, 2, 1], [2, 1, 2]])

amn0 = c.argmin(axis=0)
amx0 = c.argmax(axis=0)
amn1 = c.argmin(axis=1)
amx1 = c.argmax(axis=1)

assert amn0[0] == 0
assert amn0[1] == 1
assert amn0[2] == 0
assert amx0[0] == 1
assert amx0[1] == 0
assert amx0[2] == 1
assert amn1[0] == 0
assert amn1[1] == 1
assert amx1[0] == 1
assert amx1[1] == 0
