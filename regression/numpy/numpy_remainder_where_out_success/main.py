import numpy as np

a = np.array([7, -7, 9])
o = np.array([5, 5, 5])
r = np.remainder(a, np.array([2, 2, 4]), where=[True, False, True], out=o)
assert o[0] == 1 and o[1] == 5 and o[2] == 1
assert r[2] == 1
v = np.zeros(4)
np.floor_divide(np.array([7.0, -7.0]), 2.0, out=v[1:3])
assert v[0] == 0.0 and v[1] == 3.0 and v[2] == -4.0 and v[3] == 0.0
