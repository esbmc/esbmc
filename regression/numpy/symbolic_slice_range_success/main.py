import numpy as np

a = np.array([10, 20, 30, 40])
i = nondet_int()
j = nondet_int()
__ESBMC_assume(i >= 0 and i <= 4)
__ESBMC_assume(j >= i and j <= 4)
s = a[i:j]
assert len(s) == j - i
if len(s) > 0:
    assert s[0] == a[i]
