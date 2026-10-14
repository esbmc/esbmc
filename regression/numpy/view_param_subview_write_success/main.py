import numpy as np

def inc_tail(row):
    tail = row[1:]
    tail[0] = tail[0] + 3
    return tail[0]

a = np.array([[1, 2, 3], [4, 5, 6]])
r = inc_tail(a[0])
assert r == 5
assert a[0][1] == 5
assert a[1][1] == 5
assert a[0][0] == 1
