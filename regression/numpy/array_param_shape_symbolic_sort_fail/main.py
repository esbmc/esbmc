import numpy as np


def symbolic_sort(a):
    return a.argsort()


n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 3)
a_nondet = np.ones(n)
result = symbolic_sort(a_nondet)

assert result[0] == 0
