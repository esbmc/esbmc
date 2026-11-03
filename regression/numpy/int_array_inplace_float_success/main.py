import numpy as np

a = np.array([1, 2])
try:
    a *= 2.5
    assert False
except TypeError:
    pass
assert a[1] == 2
f = np.array([1.0, 2.0])
f *= 2
assert f[1] == 4.0
