import numpy as np
result = np.linalg.inv(np.array([[1.0, 2.0], [3.0, 4.0]]))
assert abs(result[0][0] + 2.0) < 1e-9
assert abs(result[0][1] - 1.0) < 1e-9
assert abs(result[1][0] - 1.5) < 1e-9
assert abs(result[1][1] + 0.5) < 1e-9
