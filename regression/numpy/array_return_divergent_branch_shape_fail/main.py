import numpy as np


def make(flag):
    if flag:
        return np.array([1, 2, 3])
    return np.array([1, 2, 3, 4])


def use(a):
    return a[3]


flag = nondet_int()
__ESBMC_assume(flag == 0 or flag == 1)
x = use(make(flag))
