import numpy as np

m = np.array([[1, 2, 3], [4, 5, 6]])
r = np.array([10, 20, 30])
c = np.array([[100], [200]])
s = m + r
assert s[0, 0] == 11 and s[1, 2] == 36
p = m * c
assert p[0, 1] == 200 and p[1, 2] == 1200
g = np.array([[1, 2, 3]]) + np.array([[10], [20]])
assert g[1, 0] == 21 and g[0, 2] == 13
q = m // r
assert q[1, 0] == 0
