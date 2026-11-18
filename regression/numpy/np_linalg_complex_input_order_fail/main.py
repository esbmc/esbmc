import numpy as np

# NumPy 2.5.1 returns [2+1j, 1], not the diagonal's order.
w = np.linalg.eigvals(np.array([[1.0, 0.0], [5.0, 2.0 + 1.0j]]))
assert abs(w[0] - 1.0) < 1e-6
