import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0, 30.0])
o = np.zeros(3)
np.add(a, b, where=[True, False, True], out=o)
assert o[0] == 11.0 and o[1] == 0.0 and o[2] == 33.0
p = np.full(3, 7.0)
np.subtract(b, a, where=True, out=p)
assert p[0] == 9.0 and p[1] == 18.0 and p[2] == 27.0
r = np.multiply(a, 2.0, out=o)
r[0] = 5.0
assert o[0] == 5.0 and o[2] == 6.0
q = np.zeros(6)
np.divide(b, a, out=q[::2])
assert q[0] == 10.0 and q[1] == 0.0 and q[4] == 10.0
