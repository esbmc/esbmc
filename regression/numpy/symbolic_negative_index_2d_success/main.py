import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
i = nondet_int()
__ESBMC_assume(i == -1)

assert a[i][0] == 5
assert a[i][1] == 6
