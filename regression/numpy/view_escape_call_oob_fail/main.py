import numpy as np


def at(v, i):
    return v[i]


a = np.array([10, 20, 30, 40, 50])
v = a[1:4]
k = nondet_int()
x = at(v, k)
