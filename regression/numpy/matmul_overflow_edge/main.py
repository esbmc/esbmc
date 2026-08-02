import numpy as np

A = np.matmul([[4611686018427387903, 0], [0, 1]], [[2, 0], [0, 1]])
assert A[0][0] == 9223372036854775806
assert A[0][1] == 0
assert A[1][0] == 0
assert A[1][1] == 1
