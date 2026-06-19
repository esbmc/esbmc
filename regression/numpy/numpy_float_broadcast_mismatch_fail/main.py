import numpy as np

a = np.array([[1.0, 2.0, 3.0]])
b = np.array([[4.0, 5.0], [6.0, 7.0]])
c = np.add(a, b)
assert c[0][0] == 5.0
