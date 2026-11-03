import numpy as np

a = np.array([1, 2, 3])
f = np.array([1.5, 2.0])
p = a ** 2
assert p[2] == 9
g = f ** 2
assert g[0] == 2.25
assert (a & 1)[1] == 0
assert (a | 4)[0] == 5
assert (a ^ 3)[2] == 0
assert (a << 2)[1] == 8
assert (a >> 1)[2] == 1
