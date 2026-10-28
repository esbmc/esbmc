import numpy as np

b = np.linalg.inv(np.array([[4.0, 7.0], [2.0, 6.0]]))
assert abs(b[0][0] - 0.6) < 1e-9 and abs(b[0][1] + 0.7) < 1e-9
n = np.linalg.norm(np.array([0.1, 0.2, 0.3]))
assert abs(n - 0.37416573867739417) < 1e-12
