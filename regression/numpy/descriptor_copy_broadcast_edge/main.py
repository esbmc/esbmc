import numpy as np

a = np.array([5])
b = np.broadcast_to(a, (2, 2))
c = np.copy(b)

c[0][0] = 7
a[0] = 9

assert c[0][0] == 7
assert c[1][1] == 5
assert a[0] == 9
