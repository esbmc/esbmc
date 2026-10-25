import numpy as np
result = np.linalg.solve(np.array([[2.0, 1.0], [5.0, 3.0]]), np.array([4.0, 7.0]))
assert abs(result[0] - 5.0) < 1e-9
assert abs(result[1] + 6.0) < 1e-9
