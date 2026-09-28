import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

s0 = a.sum(axis=0)
s1 = a.sum(axis=1)
m0 = a.mean(axis=0)
m1 = a.mean(axis=1)

assert s0[0] == 5
assert s0[1] == 7
assert s0[2] == 9
assert s1[0] == 6
assert s1[1] == 15
assert m0[0] == 2.5
assert m1[0] == 2.0
assert m1[1] == 5.0
