import numpy as np

a = np.array([1.0, -2.0, 3.0])

assert np.linalg.norm(a, np.inf) == 3.0
assert np.linalg.norm(a, -np.inf) == 1.0
