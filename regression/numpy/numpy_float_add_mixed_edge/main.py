import numpy as np

a = np.array([1.5, 2.5])
b = 1
c = np.add(a, b)
assert c[0] == 2.5
assert c[1] == 3.5
