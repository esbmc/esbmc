import numpy as np

x = np.array([0.0, 1.0, -2.0])
o = np.full(3, -7.0)
y = np.log1p(x, where=x > -1.0, out=o)
assert o[0] == 0.0
assert o[1] > 0.69 and o[1] < 0.70
assert o[2] == -7.0
assert y[2] == -7.0
