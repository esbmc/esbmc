import numpy as np

a = np.array([[1, 3], [4, 2]])
col = a[:, 0]  # [1, 4]

assert np.argmax(col) == 1
