import numpy as np


def symbolic_shape(a):
    return a.shape


# Trying to pass a symbolic-size array without a concrete shape
n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 3)
a_nondet = np.ones(n)
shape_result = symbolic_shape(a_nondet)

assert shape_result[0] == 3
