import numpy as np

a = np.zeros((2, 3))
assert a[0][0] == 0.0
assert a[1][2] == 0.0

a_alt = np.zeros((2, 1))
assert a_alt[0][0] == 0.0
assert a_alt[1][0] == 0.0

a_m1 = np.ones((2, 3))
assert a_m1[0][0] == 1.0
assert a_m1[1][2] == 1.0

p_m1 = np.ones((1, 2))
assert p_m1[0][0] == 1.0
assert p_m1[0][1] == 1.0

a_m2 = np.zeros((2, 3))
s_m2 = a_m2.shape
assert s_m2[0] == 2
assert s_m2[1] == 3

a_alt_m2 = np.zeros((1, 1))
s_alt_m2 = a_alt_m2.shape
assert s_alt_m2[0] == 1
assert s_alt_m2[1] == 1
