import numpy as np

a = np.array([[1.5, 2.0], [3.0, 4.5]])
b = np.array([[0.5, 2.0], [1.5, 1.5]])
c = np.divide(a, b)
assert c[0][0] == 3.0
assert c[0][1] == 1.0
assert c[1][0] == 2.0
assert c[1][1] == 3.0
