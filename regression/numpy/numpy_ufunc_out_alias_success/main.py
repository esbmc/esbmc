import numpy as np

o = np.array([1.0])
r = np.exp([0.0], out=o)
r[0] = 7.0
assert o[0] == 7.0
c = np.zeros(3)
v = np.exp([0.0, 0.0], out=c[1:])
v[0] = 5.0
assert c[1] == 5.0 and c[2] == 1.0
