import numpy as np

a = np.array([1, 2, 3])
t = np.transpose(a)

t[1] = 9

assert a[1] == 9
assert t[1] == 9
assert len(t) == 3
