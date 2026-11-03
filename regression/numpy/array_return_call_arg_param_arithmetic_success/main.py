import numpy as np


def double(a):
    return a * 2


def second_doubled(a):
    return (a * 2)[1]


def corner_doubled(m):
    return m * 2


x = np.array([1.0, 2.0])
r = double(x)
assert r[1] == 4.0
assert second_doubled(x) == 4.0

m = np.array([[1.0, 2.0], [3.0, 4.0]])
c = corner_doubled(m)
assert c[1][1] == 8.0
