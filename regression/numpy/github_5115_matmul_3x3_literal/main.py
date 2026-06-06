import numpy as np

A = np.matmul(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[9, 8, 7], [6, 5, 4], [3, 2, 1]])

assert A[0][0] == 30
assert A[1][1] == 69
assert A[2][2] == 90
