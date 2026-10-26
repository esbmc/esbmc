import numpy as np

r = np.linalg.inv(np.array([[1.0, 2.0], [3.0, 4.0]]))
assert r[0][0] == -2.0
