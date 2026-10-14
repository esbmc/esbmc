import numpy as np

a = np.array([10, 20, 30, 40])
r = a[::-1]
a = np.array([99, 99, 99, 99])

assert r[0] == 40
assert r[1] == 30
assert r[2] == 20
assert r[3] == 10
assert a[0] == 99
