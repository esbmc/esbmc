import numpy as np

a = np.array([[1.0], [2.0]])
b = np.array([3.0, 4.0])
c = np.add(a, b)
assert c[1][1] == 7.0
