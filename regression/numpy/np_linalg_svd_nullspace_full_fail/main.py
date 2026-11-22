import numpy as np

# NumPy 2.5.1 returns vh[1] = [-0.534..., 0.774..., -0.338...]: one of the
# bases of the null space, which a model fixing another basis would exclude.
u, s, vh = np.linalg.svd(np.array([[1.0, 2.0, 3.0]]))
assert abs(abs(vh[1][1]) - 0.7745419205884383) > 1e-3
