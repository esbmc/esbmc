import numpy as np

A = np.matmul([[4611686018427387903, 1], [1, 1]], [[3, 0], [0, 1]])
assert A[0][0] > 0
