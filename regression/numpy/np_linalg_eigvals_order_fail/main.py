import numpy as np

# LAPACK returns these eigenvalues as [-2, 5]: no order can be assumed.
w = np.linalg.eigvals(np.array([[1.0, 4.0], [3.0, 2.0]]))
assert w[0].real > w[1].real
