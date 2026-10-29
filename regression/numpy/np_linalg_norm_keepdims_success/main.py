import numpy as np

m = np.array([[3.0, 4.0], [6.0, 8.0]])
r = np.linalg.norm(m, axis=1, keepdims=True)
assert len(r) == 2 and len(r[0]) == 1 and abs(r[1][0] - 10.0) < 1e-9
f = np.linalg.norm(m, keepdims=True)
assert len(f) == 1 and len(f[0]) == 1 and abs(f[0][0] - 11.180339887498949) < 1e-9
