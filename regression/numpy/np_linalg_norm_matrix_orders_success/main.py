import numpy as np

q = np.array([[1.0, -2.0], [3.0, 4.0]])
assert abs(np.linalg.norm(q, 1) - 6.0) < 1e-9
assert abs(np.linalg.norm(q, -1) - 4.0) < 1e-9
assert abs(np.linalg.norm(q, np.inf) - 7.0) < 1e-9
assert abs(np.linalg.norm(q, "fro") - 5.477225575051661) < 1e-9
assert abs(np.linalg.norm(q, 2) - 5.116672736016927) < 1e-9
