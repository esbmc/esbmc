import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
print(v)
v[0] = 7
print(v)
assert a[1] == 7
