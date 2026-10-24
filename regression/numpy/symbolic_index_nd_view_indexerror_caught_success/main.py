import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
t = np.array([[1, 2], [3, 4]]).T
i = nondet_int()
__ESBMC_assume(i == 2)
caught = 0
try:
    x = v[i][0]
except IndexError:
    caught = 1
assert caught == 1
caught = 0
try:
    y = t[i][0]
except IndexError:
    caught = 1
assert caught == 1
