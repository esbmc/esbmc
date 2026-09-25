import numpy as np

p = np.percentile(np.array([1, 2, 3]), 101)

assert p == 3
