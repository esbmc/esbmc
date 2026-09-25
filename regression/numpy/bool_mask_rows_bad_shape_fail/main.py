import numpy as np

# A mask length mismatch is rejected before literal/symbolic dispatch, so
# this already holds for a symbolic (nondet-driven) mask too.
a = np.array([[1, 2], [3, 4]])
m0 = nondet_bool()
mask = np.array([m0, True, False])
b = a[mask]
