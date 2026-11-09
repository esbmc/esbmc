import numpy as np

x = np.array([0.0, 1.0, -2.0])
o = np.full(3, -7.0)
np.log1p(x, where=x > -1.0, out=o)
assert o[2] != -7.0
