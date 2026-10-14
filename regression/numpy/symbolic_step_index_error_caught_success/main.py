import numpy as np

a = np.array([10, 20, 30, 40, 50])
st = nondet_int()
__ESBMC_assume(st == 2)
s = a[::st]
caught = 0
try:
    x = s[3]
except IndexError:
    caught = 1
assert caught == 1
assert s[-3] == 10
