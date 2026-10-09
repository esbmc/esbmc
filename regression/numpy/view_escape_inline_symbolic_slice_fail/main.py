import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
i = nondet_int()
__ESBMC_assume(i >= 0 and i <= 1)
t = np.sum(a[i:, :, :])
