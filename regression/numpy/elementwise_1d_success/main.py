import numpy as np

s = np.sin([0.0, 1.0])
c = np.cos([0.0, 1.0])
e = np.exp([0.0, 1.0])
r = np.sqrt([0.0, 4.0])
a = np.arctan([0.0, 1.0])

assert s[0] == 0.0
assert s[1] > 0.84
assert c[0] == 1.0
assert c[1] > 0.53
assert e[0] == 1.0
assert e[1] > 2.71
assert r[1] == 2.0
assert a[1] > 0.78
