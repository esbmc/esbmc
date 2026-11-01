import numpy as np

a = np.zeros((2, 3))
s = a.shape
assert s[0] == 2
assert s[1] == 3

a_alt = np.zeros((1, 1))
s_alt = a_alt.shape
assert s_alt[0] == 1
assert s_alt[1] == 1
