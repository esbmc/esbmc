import numpy as np

a = np.array([1, 2, 3])
b = a + 4
c = b + 5
assert c[0] == 10
assert c[1] == 11
assert c[2] == 12
