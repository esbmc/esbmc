import numpy as np

u, s, vh = np.linalg.svd(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]), full_matrices=False)
