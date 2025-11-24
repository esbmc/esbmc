import numpy as np
a = np.array([[1, 2], [3, 4]])
x = np.linalg.det(a)
assert x == -2
#assert np.isclose(x, -2.0)
