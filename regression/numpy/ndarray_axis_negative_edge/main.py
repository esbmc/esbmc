import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

s_neg1 = a.sum(axis=-1)
s_neg2 = a.sum(axis=-2)

assert s_neg1[0] == 6
assert s_neg1[1] == 15
assert s_neg2[0] == 5
assert s_neg2[1] == 7
assert s_neg2[2] == 9
