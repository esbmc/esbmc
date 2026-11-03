import numpy as np

assert np.linalg.norm(np.array([]), axis=None) == 0.0
assert np.linalg.norm(np.array([3.0, 4.0]), axis=0) == 5.0
