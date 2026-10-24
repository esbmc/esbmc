import numpy as np

def cross_write(p, q):
    p[0] = 42
    return q[0]

a = np.array([1, 2, 3])
v = a[0:]
result = cross_write(v, v)
assert result == 42
assert a[0] == 42
