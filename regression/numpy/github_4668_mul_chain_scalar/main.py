import numpy as np

a = np.array([1, 2, 3])
b = a * 2
c = b * 3
assert c[0] == 6
assert c[1] == 12
assert c[2] == 18
