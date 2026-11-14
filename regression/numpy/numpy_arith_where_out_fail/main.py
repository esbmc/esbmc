import numpy as np

a = np.array([1.0, 2.0])
o = np.zeros(2)
np.add(a, a, where=[True, False], out=o)
assert o[1] == 4.0
