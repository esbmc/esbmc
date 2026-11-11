import numpy as np

w, v = np.linalg.eig(np.array([[3.0, 0.0], [0.0, 3.0]]))
assert w[0] == 3.0 and w[1] == 3.0
assert v[0][0] == 1.0 and v[0][1] == 0.0 and v[1][0] == 0.0 and v[1][1] == 1.0
