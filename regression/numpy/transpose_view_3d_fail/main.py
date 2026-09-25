import numpy as np

a = np.array([[[1], [2]], [[3], [4]]])
t = np.transpose(a)

assert t[0][0][0] == 1
