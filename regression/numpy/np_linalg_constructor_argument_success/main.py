import numpy as np

assert abs(np.linalg.det(np.eye(3)) - 1.0) < 1e-9
b = np.linalg.inv(np.identity(2))
assert b[0][0] == 1.0 and b[0][1] == 0.0
assert np.linalg.norm(np.ones(4)) == 2.0
