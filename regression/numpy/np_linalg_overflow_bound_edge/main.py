import numpy as np

# The largest entry to the power n overflows, the determinant does not.
d = np.linalg.det(np.array([[1e200, 0.5], [0.0, 1e-200]]))
assert abs(d - 1.0) < 1e-6
