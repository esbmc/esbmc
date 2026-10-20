import numpy as np

a = np.array([[1, 2], [3, 4]])
t = a.T
i = nondet_int()
__ESBMC_assume(i == -1)

assert t[i][0] == 2
assert t[i][1] == 4
