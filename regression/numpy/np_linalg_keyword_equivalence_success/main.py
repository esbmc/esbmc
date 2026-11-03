import numpy as np

m = np.array([[2.0, 0.0], [0.0, 4.0]])
assert abs(np.linalg.det(a=m) - 8.0) < 1e-9
assert np.linalg.norm(x=np.array([1.0, -2.0, 3.0]), ord=1) == 6.0
x = np.linalg.solve(a=m, b=np.array([2.0, 4.0]))
assert x[0] == 1.0 and x[1] == 1.0
