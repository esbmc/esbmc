import numpy as np

a = np.array([[[1], [2]], [[3], [4]]])
xs = a.tolist()

assert xs[1][1][0] == 5
