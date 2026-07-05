import numpy as np

a = np.array([[1.0, 0.0], [0.0, 1.0]])
sigma = np.linalg.svd(a)

assert sigma[0] == 1.0
assert sigma[1] == 1.0
