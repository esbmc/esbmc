import numpy as np

m = np.array([[3.0, 4.0], [6.0, 8.0]])
r = np.linalg.norm(m, axis=1)
assert len(r) == 2 and abs(r[0] - 5.0) < 1e-9 and abs(r[1] - 10.0) < 1e-9
c = np.linalg.norm(m, ord=1, axis=0)
assert abs(c[0] - 9.0) < 1e-9 and abs(c[1] - 12.0) < 1e-9
i = np.linalg.norm(m, ord=np.inf, axis=1)
assert i[0] == 4.0 and i[1] == 8.0
