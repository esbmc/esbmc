import numpy as np

a = np.array([7, -7, 5])
b = a % np.array([2, 3, -2])
assert b[0] == 1 and b[1] == 2 and b[2] == -1
s = 10 % np.array([3, 4])
assert s[0] == 1 and s[1] == 2
f = np.array([7.5, -7.5]) % 2
assert f[0] == 1.5 and f[1] == 0.5
