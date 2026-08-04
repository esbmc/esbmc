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
# var = mean((x - mean)^2) = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
assert vr > 0.6666 and vr < 0.6667
# std = sqrt(var) = sqrt(2/3)
assert sd > 0.8164 and sd < 0.8165
