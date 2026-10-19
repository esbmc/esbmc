import numpy as np

m = np.array([[1 + 2j, 3j], [1, 1]])
assert np.sum(m[0]) == 1 + 5j
