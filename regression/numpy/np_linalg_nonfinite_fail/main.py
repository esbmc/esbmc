import numpy as np

s = np.linalg.svd(np.array([[1.0, float("nan")], [0.0, 1.0]]), compute_uv=False)
