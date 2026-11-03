import numpy as np

# A triangular matrix has its diagonal as exact eigenvalues, in an order
# LAPACK's balancing decides.
w = np.linalg.eigvals(np.array([[1.0, 1.0], [0.0, 2.0]]))
assert (w[0] == 1.0 and w[1] == 2.0) or (w[0] == 2.0 and w[1] == 1.0)
v = np.linalg.eigvals(np.array([[2.0, 0.0], [7.0, 1.0]]))
assert v[0] + v[1] == 3.0
