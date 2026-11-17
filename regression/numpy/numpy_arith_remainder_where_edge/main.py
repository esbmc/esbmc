import numpy as np

a = np.array([4, 0, -8])
o = np.array([5, 5, 5])
np.reciprocal(a, where=a != 0, out=o)
assert o[0] == 0 and o[1] == 5 and o[2] == 0
k = np.array([1.5, -2.5])
np.fmod(k, 1.0, where=False, out=k)
assert k[0] == 1.5 and k[1] == -2.5
np.floor_divide(k, 1.0, where=True, out=k)
assert k[0] == 1.0 and k[1] == -3.0
