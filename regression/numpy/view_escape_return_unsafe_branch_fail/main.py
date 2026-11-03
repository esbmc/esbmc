import numpy as np


def pick(a, flag):
    if flag:
        return a[0]
    return a[:, 0]


a = np.array([[1, 2, 3], [4, 5, 6]])
c = nondet_bool()
v = pick(a, c)
x = v[0]
