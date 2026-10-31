import numpy as np

u, s, vh = np.linalg.svd(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]]))
