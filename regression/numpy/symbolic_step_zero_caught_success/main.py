import numpy as np

a = np.array([10, 20, 30])
st = nondet_int()
__ESBMC_assume(st == 0)
caught = 0
try:
    s = a[::st]
except ValueError:
    caught = 1
assert caught == 1
