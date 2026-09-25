import numpy as np

a = np.array([[[9]]])
b = a[:, 0, 0]

assert b[0] == 9
