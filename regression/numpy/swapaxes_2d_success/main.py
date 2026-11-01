import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.swapaxes(a, 0, 1)

t[1][0] = 7

assert a[0][1] == 7
assert t[1][0] == 7

p = np.array([[1, 2], [3, 4]])
s = np.swapaxes(p, 1, 1)

s[0][1] = 6

assert p[0][1] == 6
assert s[0][1] == 6

a_m1 = np.array([[1, 2], [3, 4]])
t_m1 = np.moveaxis(a_m1, 0, 1)

t_m1[0][1] = 7

assert a_m1[1][0] == 7
assert t_m1[0][1] == 7

p_m1 = np.array([[1, 2], [3, 4]])
s_m1 = np.moveaxis(p_m1, 1, 1)

s_m1[0][1] = 6

assert p_m1[0][1] == 6
assert s_m1[0][1] == 6
