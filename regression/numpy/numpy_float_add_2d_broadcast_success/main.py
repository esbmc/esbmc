import numpy as np

a = np.array([[1.0], [2.0]])
b = np.array([3.0, 4.0])
c = np.add(a, b)
assert c[0][0] == 4.0
assert c[0][1] == 5.0
assert c[1][0] == 5.0
assert c[1][1] == 6.0
