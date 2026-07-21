import numpy as np

# ADR-NP-001: a symbolic 2-D row mask is now supported through the canonical
# bounded ndarray descriptor result (a buffer sized at the worst case plus a
# runtime logical row count), see numpy-architecture-decisions.md.
a = np.array([[1, 2], [3, 4]])
n = nondet_bool()
mask = np.array([n, not n])
x = a[mask]
