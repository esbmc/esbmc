import numpy as np

a = np.array([[8.0, 6.0], [9.0, 12.0]])
b = np.array([[2.0, 3.0], [3.0, 4.0]])
c = np.divide(a, b)
assert c[0][0] == 4.0
assert c[0][1] == 2.0
assert c[1][0] == 3.0
assert c[1][1] == 3.0

a_alt = np.array([[1.5, 2.0], [3.0, 4.5]])
b_alt = np.array([[0.5, 2.0], [1.5, 1.5]])
c_alt = np.divide(a_alt, b_alt)
assert c_alt[0][0] == 3.0
assert c_alt[0][1] == 1.0
assert c_alt[1][0] == 2.0
assert c_alt[1][1] == 3.0
