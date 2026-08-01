import numpy as np

assert np.percentile(np.array([1, 2, 3, 4, 5]), 0) == 1
assert np.percentile(np.array([1, 2, 3, 4, 5]), 25) == 2
assert np.percentile(np.array([1, 2, 3, 4, 5]), 50) == 3
assert np.percentile(np.array([1, 2, 3, 4, 5]), 75) == 4
assert np.percentile(np.array([1, 2, 3, 4, 5]), 100) == 5
