import numpy as np

# True zero-length bool arrays (np.array([], dtype=bool)) are not yet
# constructible by the frontend, so this exercises the smallest boundary
# case instead: a single-element array filtered down to an empty result.
a = np.array([7])
mask = np.array([False])
b = a[mask]

assert len(b) == 0
