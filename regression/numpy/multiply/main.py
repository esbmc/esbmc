import numpy as np

a = np.multiply(2.0, 4.0)
assert a == 8.0

b = np.multiply([3,4],[1,3])
assert b[0] == 3
assert b[1] == 12

c = np.array([5,4])
d = np.array([2,2])
e = np.multiply(c, d)
assert e[0] == 10
assert e[1] == 8