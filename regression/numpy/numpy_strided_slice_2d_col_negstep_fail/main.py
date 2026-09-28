import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b = a[:, ::-2]

assert b[0][0] == 4
