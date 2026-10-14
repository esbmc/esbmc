import numpy as np

a = np.array([1.5, 2.5])
b = np.array([1, 2])
c = np.add(a, b)
assert c[0] == 2.5
assert c[1] == 4.5
