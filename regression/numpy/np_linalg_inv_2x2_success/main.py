import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0]])
inv = np.linalg.inv(A)
assert inv[0][0] == -2.0
assert inv[0][1] == 1.0
assert inv[1][0] == 1.5
assert inv[1][1] == -0.5
