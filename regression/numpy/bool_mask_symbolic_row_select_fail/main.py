import numpy as np

# ADR-NP-001: stays explicitly rejected until the canonical ndarray
# descriptor exists (see numpy-architecture-decisions.md) - a
# bounded-capacity result with a detached `count` variable is not sound,
# since len()/shape() would observe physical capacity, not logical size.
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
n = nondet_bool()
mask = np.array([n, not n, True])
b = a[mask]
