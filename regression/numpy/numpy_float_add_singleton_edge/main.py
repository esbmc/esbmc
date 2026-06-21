import numpy as np

a = np.array([[1.5], [2.5]])
b = np.array([10.0, 20.0])
c = np.add(a, b)
assert c[0][0] == 11.5
assert c[0][1] == 21.5
assert c[1][0] == 12.5
assert c[1][1] == 22.5
