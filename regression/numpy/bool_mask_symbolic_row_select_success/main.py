import numpy as np

# ADR-NP-001: now supported through the canonical bounded ndarray descriptor
# result (see numpy-architecture-decisions.md).
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
n = nondet_bool()
mask = np.array([n, not n, True])
b = a[mask]
