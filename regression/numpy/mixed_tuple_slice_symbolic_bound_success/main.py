import numpy as np

n = nondet_int()
__ESBMC_assume(n >= 1 and n <= 5)
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = a[0:n, 0, 0]
assert b[0] == 1
if n >= 2:
    assert b[1] == 5
