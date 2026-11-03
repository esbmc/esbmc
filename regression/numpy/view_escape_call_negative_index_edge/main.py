import numpy as np


def last(v):
    return v[-1]


def at(v, i):
    return v[i]


a = np.array([10, 20, 30, 40, 50])
v = a[1:4]
assert last(v) == 40
k = nondet_int()
__ESBMC_assume(k == -3)
assert at(v, k) == 20
