import numpy as np

A = np.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])

assert A[0][0] == 19
assert A[0][1] == 22
assert A[1][0] == 43
assert A[1][1] == 50
