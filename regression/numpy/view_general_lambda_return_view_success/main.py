import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
tail = lambda w: w[1:]
t = tail(v)
t[0] = 9
assert a[2] == 9
assert len(t) == 2
