import numpy as np

a = np.array([])
scalar_idx = np.searchsorted(a, 3)
vector_idx = np.searchsorted(a, [3, 5])

assert scalar_idx == 0
assert vector_idx[0] == 0
assert vector_idx[1] == 0
