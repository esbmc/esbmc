import numpy as np

a = np.array([0.0, 1.0, 2.0])
b = np.exp(a + np.array([[0.0], [1.0]]))
assert b[0][0] == 1.0 and b[1][0] > 2.71 and b[0][2] > 7.38
