import numpy as np

C = np.dot([[1, 2], [3, 4]], [[5, 6], [7, 8]])

assert C[0][0] == 19
assert C[0][1] == 22
assert C[1][0] == 43
assert C[1][1] == 50
