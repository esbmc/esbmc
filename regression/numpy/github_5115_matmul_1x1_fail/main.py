import numpy as np

a = nondet_int()
c = nondet_int()
__ESBMC_assume(a >= -5 and a <= 5)
__ESBMC_assume(c >= -5 and c <= 5)

A = np.matmul([[a]], [[c]])

assert A[0][0] == a * c + 1
