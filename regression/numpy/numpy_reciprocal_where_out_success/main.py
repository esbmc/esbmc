import numpy as np

a = np.array([2.0, 0.0, 4.0])
o = np.array([9.0, 9.0, 9.0])
np.reciprocal(a, where=a != 0.0, out=o)
assert o[0] == 0.5 and o[1] == 9.0 and o[2] == 0.25
p = np.zeros(3)
np.float_power(np.array([2, 3]), 2, out=p[1:])
assert p[0] == 0.0 and p[1] == 4.0 and p[2] == 9.0
