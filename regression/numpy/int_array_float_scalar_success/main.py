import numpy as np

a = np.array([1, 2])
b = a * 2.5
assert b[0] == 2.5 and b[1] == 5.0
c = a + 0.5
assert c[1] == 2.5
assert a[1] == 2
