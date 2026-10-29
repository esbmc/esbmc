import numpy as np

a = np.array([[2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 5.0]])
w, v = np.linalg.eig(a)
assert w[0] == 2.0 and w[1] == -1.0 and w[2] == 5.0
assert v[0][0] == 1.0 and v[1][1] == 1.0 and v[2][2] == 1.0
assert v[0][1] == 0.0 and v[2][0] == 0.0
