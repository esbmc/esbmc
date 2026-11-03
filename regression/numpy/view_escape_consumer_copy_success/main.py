import numpy as np


def first_row(a):
    return a[0]


a = np.array([[1, 2, 3], [4, 5, 6]])
r = first_row(a)
c1 = np.copy(r)
c2 = r.copy()
c3 = np.array(r)
a[0][0] = 9
assert r[0] == 9
assert c1[0] == 1
assert c2[0] == 1
assert c3[0] == 1

box = [a[1]]
c4 = np.copy(box[0])
a[1][0] = 7
assert c4[0] == 4
