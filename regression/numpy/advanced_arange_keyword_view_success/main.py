import numpy as np

a = np.arange(stop=6)
v = a[1:4]
a[2] = 9
assert v[1] == 9
v[0] = -1
assert a[1] == -1
