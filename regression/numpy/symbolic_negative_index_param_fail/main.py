import numpy as np


def check_last_first(a):
    i = nondet_int()
    __ESBMC_assume(i == -1)
    assert a[i][0] == 1


a = np.array([[1, 2], [3, 4], [5, 6]])
check_last_first(a)
