import numpy as np


def first_of_row(a, i: int):
    return a[i][0]


a = np.array([[1, 2], [3, 4], [5, 6]])
k = nondet_int()
__ESBMC_assume(k == -1)
assert first_of_row(a, k) == 5
