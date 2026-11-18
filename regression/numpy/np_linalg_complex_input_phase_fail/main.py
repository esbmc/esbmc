import numpy as np

# LAPACK makes each eigenvector's largest component real: here both are in
# the second row.
w, v = np.linalg.eig(np.array([[1.0, 0.0], [5.0, 2.0 + 1.0j]]))
assert abs(v[1][0].imag) + abs(v[1][1].imag) > 1e-3
