import numpy as np

a = np.array([10, 20, 30, 40])
i = nondet_int()
j = nondet_int()
k = nondet_int()
__ESBMC_assume(i == 1)
__ESBMC_assume(j == 3)
__ESBMC_assume(k == 2)
s = a[i:j]
caught = 0
try:
    x = s[k]
except IndexError:
    caught = 1
assert caught == 1
assert s[-1] == 30
