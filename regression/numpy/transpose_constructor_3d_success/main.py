import numpy as np

# transpose now materializes a rank-3 axis permutation from a constructor
# array instead of rejecting shapes beyond 2D.
a = np.zeros((2, 2, 2))
b = np.transpose(a)

assert b[0][0][0] == 0
