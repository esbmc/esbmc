import numpy as np

a = np.array([[1.0], [2.0]])
b = np.array([3.0, 4.0])
c = np.add(a, b)
assert c[0][0] == 4.0
assert c[0][1] == 5.0
assert c[1][0] == 5.0
assert c[1][1] == 6.0

a_alt = np.array([[1.5], [2.5]])
b_alt = np.array([10.0, 20.0])
c_alt = np.add(a_alt, b_alt)
assert c_alt[0][0] == 11.5
assert c_alt[0][1] == 21.5
assert c_alt[1][0] == 12.5
assert c_alt[1][1] == 22.5
