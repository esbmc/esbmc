import numpy as np

a = np.array([1.5, 2.5])
b = np.array([3.0, 4.0])
c = np.add(a, b)
assert c[0] == 4.5
assert c[1] == 6.5
