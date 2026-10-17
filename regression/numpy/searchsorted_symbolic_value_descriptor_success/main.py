import numpy as np


def make():
    return np.array([1, 3, 5])


x = nondet_int()
__ESBMC_assume(x >= 2 and x < 3)

idx = np.searchsorted(make(), x)
assert idx == 1
