import numpy as np

a = np.array([[1, 2], [3, 4]])
i = nondet_int()
__ESBMC_assume(i >= 0 and i < 2)
row = a[i]

assert row[0] == a[i][0]
assert row[1] == a[i][1]
