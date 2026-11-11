import numpy as np

m = np.array([[2.0, 1.0, 9.0], [1.0, 3.0, 9.0]])
w = np.linalg.eigvals(a=m[0:2, 0:2])
assert abs(w[0].real + w[1].real - 5.0) < 1e-9
vals, vecs = np.linalg.eig(a=np.array([[4.0, 1.0, 2.0], [0.5, 3.0, 1.0], [1.0, 0.0, 1.0]]))
assert abs(vals[0] + vals[1] + vals[2] - 8.0) < 1e-9
s = np.linalg.svd(a=np.identity(3), compute_uv=False)
assert s[0] == 1.0 and s[2] == 1.0
