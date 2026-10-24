import numpy as np

m = np.array([[1, 2], [3, 4]])
d = m - 1
assert d[1][1] == 3 and d[0][0] == 0
t = m * 2
assert t[1][0] == 6
h = m / 2
assert h[0][1] == 1.0
s = m + m
assert s[1][1] == 8
q = m // 3
assert q[1][1] == 1
