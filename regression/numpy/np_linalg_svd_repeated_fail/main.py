import numpy as np

# The singular value 6 is double: NumPy 2.5.1 returns vh[0] = [0, -1, 0],
# and any rotation of that plane, the same in U and in Vh, may take its
# place.
a = np.array([[4.0, 4.0, 1.0], [-4.0, 2.0, 2.0], [2.0, -4.0, 2.0]])
u, s, vh = np.linalg.svd(a)
assert abs(abs(vh[0][1]) - 1.0) > 1e-3
