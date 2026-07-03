import numpy as np

m = np.array([[1, 2], [3, 4]])
assert m[:, 0][0] == 1
assert m[:, 0][1] == 3
