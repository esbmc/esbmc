import numpy as np

a = np.array([[1, 2], [3, 4]])
c = a[:, 1]
assert c[0] == 1
