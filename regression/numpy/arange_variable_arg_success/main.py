import numpy as np

# A Name argument bound to a constant (not a function parameter) must still
# be fast-path materialized: resolve it to its declared value first.
n = 3
a = np.arange(n)
assert len(a) == 3
assert a[2] == 2
