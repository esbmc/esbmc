import numpy as np

x = np.array([1.0, 1.0])
o = np.zeros(2)
np.add(x, x, where=[True, False], out=o)
assert o[1] == 2.0
