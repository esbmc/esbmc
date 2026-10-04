import numpy as np

a = np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
b = a[:, :, 1]
i = nondet_int()
__ESBMC_assume(i == 2)
caught = 0
try:
    x = b[i][0]
except IndexError:
    caught = 1
assert caught == 1
assert b[-2][0] == 1
