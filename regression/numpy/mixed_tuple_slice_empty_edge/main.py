import numpy as np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = a[2:1, 0, 0]

assert len(b) == 0
