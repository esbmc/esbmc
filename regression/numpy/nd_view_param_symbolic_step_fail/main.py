import numpy as np


def flip(m):
    st = nondet_int()
    __ESBMC_assume(st == -1)
    r = m[::st]
    return r[0][1]


a = np.array([[1, 2], [3, 4], [5, 6]])
assert flip(a) == 5
