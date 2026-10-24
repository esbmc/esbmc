import numpy as np


def row_head(a):
    i = nondet_int()
    __ESBMC_assume(i >= 0 and i <= 2)
    r = a[i]
    return r[0] - 2 * i


a = np.array([[1, 2], [3, 4], [5, 6]])
assert row_head(a) == 3
