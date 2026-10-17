import numpy as np

n = nondet_int()
__ESBMC_assume(1 <= n and n <= 2)
a = np.array([1, 2, 3, 4])
x = a[::n]
y = a[::n]
a = np.array([9, 9, 9, 9])
x[0] = 7
z = np.sort(y)
