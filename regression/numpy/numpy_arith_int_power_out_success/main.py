import numpy as np

a = np.array([2, 3])
o = np.array([7, 7])
np.power(a, np.array([-1, 2]), where=[False, True], out=o)
assert o[0] == 7 and o[1] == 9
np.power(a, 3, out=o)
assert o[0] == 8 and o[1] == 27
w = np.zeros(1, dtype=np.int64)
np.power(np.array([2 ** 40]), 2, out=w)
assert w[0] == 0
f = np.zeros(2)
np.power(a, 2, out=f)
assert f[0] == 4.0 and f[1] == 9.0
