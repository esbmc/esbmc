import numpy as np

q = np.array([[1.0, -2.0], [3.0, 4.0]])
assert np.linalg.norm(q, -np.inf) == 3.0
r = np.linalg.norm(q, ord=-np.inf, axis=0)
assert r[0] == 1.0 and r[1] == 2.0
