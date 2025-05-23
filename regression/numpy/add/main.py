import numpy as np

x = 2
y = 3
z = np.add(x,y)
assert z == 5

a = np.add(1.0, 4.0)
assert a == 5.0

b = np.add(1, 2)
assert b == 3

c = np.add(127, 1, dtype=np.int8)
assert c == -128

d = np.array([1,2])
e = np.array([3,4])
f = np.add(d, e)
assert f[0] == 4
assert f[1] == 6

g = np.add([1,2],[3,4])
assert g[0] == 4
assert g[1] == 6