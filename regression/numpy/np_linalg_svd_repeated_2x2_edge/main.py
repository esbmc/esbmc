import numpy as np

# Both singular values are sqrt(13); their computed gap rounds to -4.4e-16,
# and the vectors may be any rotation of the plane.
u, s, vh = np.linalg.svd(np.array([[3.0, 2.0], [2.0, -3.0]]))
assert abs(s[0] - 3.605551275463989) < 1e-9 and abs(s[0] - s[1]) < 1e-9
assert abs(vh[0][0]) <= 1.0 + 1e-9 and abs(u[1][1]) <= 1.0 + 1e-9
