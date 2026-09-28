import numpy as np

c = np.array([[3, 1, 2], [6, 4, 5]])
amn0 = np.argmin(c, axis=0)

assert amn0[1] == 1
