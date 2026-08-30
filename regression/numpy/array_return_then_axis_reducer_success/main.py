import numpy as np


def make():
    return np.array([[1, 2, 3], [4, 5, 6]])


y = make()
s0 = y.sum(axis=0)
s1 = y.sum(axis=1)

assert s0[0] == 5
assert s0[1] == 7
assert s0[2] == 9
assert s1[0] == 6
assert s1[1] == 15
