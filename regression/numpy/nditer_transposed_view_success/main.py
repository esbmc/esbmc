import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
t = a.T
it = np.nditer(t, flags=["c_index"])
seen = []
while not it.finished:
    seen.append((int(it[0]), it.index))
    it.iternext()
assert seen == [(1, 0), (2, 2), (3, 4), (4, 1), (5, 3), (6, 5)]
for x in np.nditer(t, op_flags=[["readwrite"]]):
    x[...] = x * 10
assert a[0][1] == 20
assert t[2][1] == 60
