import numpy as np
m = np.zeros((2, 3))
np.exp(np.array([0.0, 0.0]), out=m[:, 1])
assert m[0][1] == 1.0 and m[1][1] == 1.0
assert m[0][0] == 0.0 and m[1][2] == 0.0
c = np.full(4, 7.0)
np.sqrt(np.array([4.0, 9.0]), where=[False, True], out=c[::2])
assert c[0] == 7.0 and c[2] == 3.0 and c[1] == 7.0
a = np.array([0.0, 1.0, 2.0])
np.exp(a[:2], out=a[1:])
assert a[0] == 0.0 and a[1] == 1.0
assert a[2] > 2.71 and a[2] < 2.72
