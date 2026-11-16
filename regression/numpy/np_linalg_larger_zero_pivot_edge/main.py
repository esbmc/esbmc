import numpy as np

# A zero in the first pivot position takes a row swap; the factors stay exact.
p = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
assert np.linalg.det(p) == -1.0
q = np.linalg.inv(p)
assert q[0][1] == 1.0 and q[1][0] == 1.0 and q[0][0] == 0.0 and q[3][3] == 1.0
