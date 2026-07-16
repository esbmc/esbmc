import numpy as np

# ADR-NP-001: a symbolic 2-D row mask (nondet-derived) stays explicitly
# rejected until the canonical ndarray descriptor exists to carry the
# logical row count as part of the value itself.
n = nondet_bool()
__ESBMC_assume(n)

a = np.array([[1, 2], [3, 4]])
mask = np.array([n, not n])
b = a[mask]
