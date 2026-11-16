import numpy as np

a = np.array([2.0, 3.0])
o = np.zeros(2)
np.true_divide(a, 2.0, where=[False, True], out=o)
assert o[0] == 0.0 and o[1] == 1.5
s = np.square(a)
assert s[0] == 4.0 and s[1] == 9.0
np.square(a, out=o)
assert o[1] == 9.0
