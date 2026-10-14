import numpy as np

values = np.unique(np.array([1, 1]), return_counts=True)

assert len(values) == 1
