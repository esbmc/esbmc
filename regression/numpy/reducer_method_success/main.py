import numpy as np

a = np.array([1.0, 2.0, 3.0])

s = a.sum()
m = a.mean()
mn = a.min()
mx = a.max()
sd = a.std()
vr = a.var()

assert s == 6.0
assert m == 2.0
assert mn == 1.0
assert mx == 3.0
assert sd >= 0.0
assert vr >= 0.0
