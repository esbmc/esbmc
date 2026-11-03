import numpy as np

a = np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [0.0, 0.0, 1.0, 4.0]])
x = np.linalg.solve(a, np.array([5.0, 6.0, 6.0, 5.0]))
assert abs(x[0] - 1.0) < 1e-9 and abs(x[1] - 1.0) < 1e-9
assert abs(x[2] - 1.0) < 1e-9 and abs(x[3] - 1.0) < 1e-9
