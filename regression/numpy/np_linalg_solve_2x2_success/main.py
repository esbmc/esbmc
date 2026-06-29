import numpy as np
A = np.array([[2.0, 1.0], [5.0, 3.0]])
b = np.array([4.0, 7.0])
x = np.linalg.solve(A, b)
assert x[0] == 5.0
assert x[1] == -6.0
