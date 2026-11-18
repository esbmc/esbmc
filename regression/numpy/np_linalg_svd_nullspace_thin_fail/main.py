import numpy as np

# NumPy 2.5.1 returns vh[1] = [0.944..., -0.308..., -0.109...] for the zero
# singular value: any unit vector of the null space may take its place.
c = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
u, s, vh = np.linalg.svd(c, full_matrices=False)
assert abs(abs(vh[1][0]) - 0.9449111825230679) > 1e-3
