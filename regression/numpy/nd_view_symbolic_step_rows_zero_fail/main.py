import numpy as np

a = np.array([[0, 1], [2, 3], [4, 5]])
st = nondet_int()
__ESBMC_assume(st == 0)
r = a[::st]
