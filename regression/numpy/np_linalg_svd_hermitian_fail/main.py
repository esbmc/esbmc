import numpy as np

s = np.linalg.svd(np.array([[2.0, 1.0], [1.0, 2.0]]), compute_uv=False, hermitian=True)
