import numpy as np

z = np.linalg.norm(np.array([[0.0, 0.0], [0.0, 0.0]]), axis=0)
assert z[0] == 0.0 and z[1] == 0.0
assert np.linalg.norm(np.array([[0.0, 0.0], [0.0, 0.0]]), -np.inf) == 0.0
