import numpy as np

# The eigenvalue 1 is double: two of the three are within rounding of it.
w = np.linalg.eigvals(np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]))
near = 0
for k in range(3):
    if abs(w[k].real - 1.0) < 1e-6:
        near += 1
assert near == 1
