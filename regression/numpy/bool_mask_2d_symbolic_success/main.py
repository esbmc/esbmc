import numpy as np

# A symbolic 2-D row mask now resolves via the bounded-result + explicit-
# count model instead of being rejected (see numpy-architecture-decisions.md
# #1); the mask's own solved value decides which row lands at x[0].
a = np.array([[1, 2], [3, 4]])
n = nondet_bool()
mask = np.array([n, not n])
x = a[mask]
