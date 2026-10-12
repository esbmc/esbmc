import numpy as np

def first(row):
    n = len(row)
    return row[0] + n

a = np.array([[1, 2, 3], [4, 5, 6]])
flag = True
if flag:
    v = a[0]
else:
    v = a[:, 0]
r = first(v)
assert r == 4
