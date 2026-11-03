import numpy as np

q = np.array([7, -7]) // 2
assert q[0] == 3 and q[1] == -4
z = q // 0
assert z[0] == 0 and z[1] == 0
f = np.array([1.0, 2.0]) // 0.5
assert f[1] == 4.0
m = np.array([5, -5]) % 3
assert m[0] == 2 and m[1] == 1
