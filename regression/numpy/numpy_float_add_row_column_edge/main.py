import numpy as np

a = np.array([[1.0, 2.0, 3.0]])
b = np.array([[10.0], [20.0]])
c = np.add(a, b)
assert c[0][0] == 11.0
assert c[0][1] == 12.0
assert c[0][2] == 13.0
assert c[1][0] == 21.0
assert c[1][1] == 22.0
assert c[1][2] == 23.0
