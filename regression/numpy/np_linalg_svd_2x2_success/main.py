import numpy as np

a = np.array([[3.0, 0.0], [0.0, 2.0]])
sigma = np.linalg.svd(a)

assert sigma[0] == 3.0
assert sigma[1] == 2.0
