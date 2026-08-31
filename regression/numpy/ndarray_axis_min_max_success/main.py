import numpy as np

a = np.array([[3, -1, 2], [6, 4, -5]])

mn0 = a.min(axis=0)
mn1 = a.min(axis=1)
mx0 = a.max(axis=0)
mx1 = a.max(axis=1)

assert mn0[0] == 3
assert mn0[1] == -1
assert mn0[2] == -5
assert mn1[0] == -1
assert mn1[1] == -5
assert mx0[0] == 6
assert mx0[1] == 4
assert mx0[2] == 2
assert mx1[0] == 3
assert mx1[1] == 6
