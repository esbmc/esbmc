import numpy as np
nearly_singular = np.array([[1.0, 2.0], [2.0, 4.000000001]])
det_small = np.linalg.det(nearly_singular)
assert abs(det_small) < 1e-8
