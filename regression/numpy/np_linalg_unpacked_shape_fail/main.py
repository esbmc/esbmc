import numpy as np

u, s, vh = np.linalg.svd(np.array([[3.0, 4.0]]))
assert vh.shape == (1, 2)
