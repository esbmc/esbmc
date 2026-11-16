import numpy as np

a = np.array([4.0, 0.0])
o = np.array([5.0, 5.0])
np.reciprocal(a, where=a != 0.0, out=o)
assert o[1] == np.inf
