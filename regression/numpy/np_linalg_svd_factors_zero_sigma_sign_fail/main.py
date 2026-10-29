import numpy as np

# With s[1] == 0, u's second column and vh's second row take their own signs:
# NumPy gives u[1][1] * vh[1][0] = +0.986.
u, s, vh = np.linalg.svd(np.array([[-1.0, 6.0], [0.0, 0.0]]))
assert u[1][1] * vh[1][0] < 0.0, "sign"
