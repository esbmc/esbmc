import numpy as np

a = np.array([10, 20, 30, 40, 50])
st = nondet_int()
__ESBMC_assume(st == 2)
s = a[::st]
total = 0
for x in np.nditer(s):
    total += x
assert total == 91
