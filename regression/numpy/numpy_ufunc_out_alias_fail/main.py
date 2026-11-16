import numpy as np

o = np.array([1.0])
r = np.exp([0.0], out=o)
r[0] = 7.0
assert o[0] == 1.0
