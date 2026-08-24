import numpy as np

a = np.array([[[1, 2], [3, 4]]])
t = a.transpose((0, 2, 1))

assert t[0][0][0] == 1
