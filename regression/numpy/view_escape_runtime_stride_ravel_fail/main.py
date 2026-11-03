import numpy as np


def flat(v):
    return np.ravel(v)


a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
s = nondet_int()
__ESBMC_assume(s >= 1 and s <= 2)
v = a[:, ::s]
f = flat(v)
x = f[0]
