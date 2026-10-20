import numpy as np

a = np.array([[[1], [2]], [[3], [4]]])
xs = a.tolist()

assert xs[0][0][0] == 1
assert xs[0][1][0] == 2
assert xs[1][0][0] == 3
assert xs[1][1][0] == 4
