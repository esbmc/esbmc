import numpy as np

x = np.array([0.0, 1.0])
o = np.zeros(2)
np.exp(x, o)
assert o[0] == 0.0
