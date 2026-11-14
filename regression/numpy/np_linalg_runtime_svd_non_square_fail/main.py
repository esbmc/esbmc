import numpy as np

x = nondet_float()
s = np.linalg.svd(np.array([[x, 1.0, 2.0], [3.0, 4.0, 5.0]]), compute_uv=False)
