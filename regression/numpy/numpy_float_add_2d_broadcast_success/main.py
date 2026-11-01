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

a_m1 = np.array([[8.0, 6.0], [9.0, 12.0]])
b_m1 = np.array([[2.0, 3.0], [3.0, 4.0]])
c_m1 = np.divide(a_m1, b_m1)
assert c_m1[0][0] == 4.0
assert c_m1[0][1] == 2.0
assert c_m1[1][0] == 3.0
assert c_m1[1][1] == 3.0

a_alt_m1 = np.array([[1.5, 2.0], [3.0, 4.5]])
b_alt_m1 = np.array([[0.5, 2.0], [1.5, 1.5]])
c_alt_m1 = np.divide(a_alt_m1, b_alt_m1)
assert c_alt_m1[0][0] == 3.0
assert c_alt_m1[0][1] == 1.0
assert c_alt_m1[1][0] == 2.0
assert c_alt_m1[1][1] == 3.0

a_m2 = np.array([1.5, 2.5])
b_m2 = np.array([3.0, 4.0])
c_m2 = np.add(a_m2, b_m2)
assert c_m2[0] == 4.5
assert c_m2[1] == 6.5

p_m2 = np.array([1.5, 2.5])
q_m2 = np.array([1, 2])
r_m2 = np.add(p_m2, q_m2)
assert r_m2[0] == 2.5
assert r_m2[1] == 4.5
