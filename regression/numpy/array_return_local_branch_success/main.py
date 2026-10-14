import numpy as np


def make(flag):
    if flag:
        a = np.zeros(3)
        a[0] = 1
    else:
        a = np.zeros(3)
        a[0] = 2
    return a


flag = nondet_int()
__ESBMC_assume(flag == 0 or flag == 1)
y = make(flag)

assert y[0] == 1 or y[0] == 2
assert y[1] == 0
