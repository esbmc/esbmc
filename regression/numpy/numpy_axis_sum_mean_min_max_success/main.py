import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

s0 = np.sum(a, axis=0)
s1 = np.sum(a, axis=1)
m0 = np.mean(a, axis=0)
m1 = np.mean(a, axis=1)
mn0 = np.min(a, axis=0)
mx1 = np.max(a, axis=1)

assert s0[0] == 5
assert s0[1] == 7
assert s0[2] == 9
assert s1[0] == 6
assert s1[1] == 15
assert m0[0] == 2.5
assert m1[0] == 2.0
assert m1[1] == 5.0
assert mn0[0] == 1
assert mn0[1] == 2
assert mn0[2] == 3
assert mx1[0] == 3
assert mx1[1] == 6
