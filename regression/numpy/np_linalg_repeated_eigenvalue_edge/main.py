import numpy as np

w = np.linalg.eigvals(np.array([[3.0, 1.0], [0.0, 3.0]]))
assert w[0] == 3.0 and w[1] == 3.0
u, s, vh = np.linalg.svd(np.array([[2.0, 0.0], [0.0, 2.0]]))
assert s[0] == 2.0 and s[1] == 2.0
