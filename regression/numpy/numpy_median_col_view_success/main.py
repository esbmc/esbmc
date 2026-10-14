import numpy as np

a = np.array([[1, 3], [2, 4]])
col = a[:, 1]  # [3, 4]

assert np.median(col) == 3.5
