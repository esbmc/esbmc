import numpy as np

a = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
w, v = np.linalg.eig(np.array([[2.0, 0.0], [0.0, 3.0]]))
u, s, vh = np.linalg.svd(a)
p, q, r = np.linalg.svd(a, full_matrices=False)
assert w.shape == (2,) and v.shape == (2, 2)
assert u.shape == (2, 2) and s.shape == (2,) and vh.shape == (3, 3)
assert p.shape == (2, 2) and q.shape == (2,) and r.shape == (2, 3)
assert v.ndim == 2 and s.ndim == 1 and vh.size == 9 and r.size == 6
x = vh
assert x.shape == (3, 3)
