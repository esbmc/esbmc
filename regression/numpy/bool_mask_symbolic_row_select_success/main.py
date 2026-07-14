import numpy as np

# A symbolic mask spanning three rows also resolves via the bounded-result +
# explicit-count model (numpy-architecture-decisions.md #1) instead of being
# rejected - the last row is always selected regardless of n's value.
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
n = nondet_bool()
mask = np.array([n, not n, True])
b = a[mask]
