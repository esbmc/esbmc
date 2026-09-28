import numpy as np


def symbolic_len(a):
    return len(a)


n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 3)
a_nondet = np.ones(n)
result = symbolic_len(a_nondet)

assert result == n
