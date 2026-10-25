import numpy as np

sigma = np.linalg.svd(np.array([[3.0, 0.0], [0.0, 2.0]]))
assert sigma[0] == 3.0
