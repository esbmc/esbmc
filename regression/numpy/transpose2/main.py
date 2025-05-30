import numpy as np
a = np.array([[5]])
b = np.transpose(a)
assert b[0][0] == 5
