import numpy as np


def symbolic_transpose(a):
    return np.transpose(a)


n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 3)
a_nondet = np.ones((n, 2))
result = symbolic_transpose(a_nondet)

assert result.shape[0] == 2
