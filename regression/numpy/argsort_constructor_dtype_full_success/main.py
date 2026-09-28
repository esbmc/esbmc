import numpy as np

# Without the dtype=int truncation, linspace(2.9, 2.0, 3) is strictly
# decreasing, so argsort would reorder to [2, 1, 0]. Truncated to int every
# element becomes 2, so a stable sort keeps the original order instead.
idx = np.argsort(np.linspace(2.9, 2.0, 3, dtype=int))

assert idx[0] == 0
assert idx[1] == 1
assert idx[2] == 2
