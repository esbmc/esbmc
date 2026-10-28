import numpy as np

b = np.linalg.inv(np.array([[1.0, 1.0], [1.0, 1.0000000001]]))
assert b[0][0] > 1e9
