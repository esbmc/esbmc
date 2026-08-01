import numpy as np

n = int(input())
x = np.linalg.solve(np.array([[n, 0], [0, 1]]), np.array([1, 1]))

assert x[0] == 1
