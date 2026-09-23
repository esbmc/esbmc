import numpy as np

m = np.mean(np.eye(3, dtype=int))

assert abs(m - (1.0 / 3.0)) < 1e-6
