import numpy as np

m = np.array([[-1.57, 0.27, -0.78], [-1.57, 0.27, -0.78], [0.62, 0.75, -2.61]])
assert np.linalg.det(m) != 0.0
