import numpy as np

a = np.array([[8.0, 6.0], [9.0, 12.0]])
b = np.array([[2.0, 3.0], [3.0, 4.0]])
c = np.divide(a, b)
assert c[1][1] == 4.0
