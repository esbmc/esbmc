import numpy as np

r = np.array([1, 4]) ** 0.5
assert r[0] == 1.0 and r[1] == 2.0
f = np.array([7, 8]) // 2.0
assert f[0] == 3.0 and f[1] == 4.0
m = np.array([7, 8]) % 2.5
assert m[0] == 2.0 and m[1] == 0.5
