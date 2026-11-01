import numpy as np

a = np.array([1, 2, 3])
b = a + 100

assert b[0] == 101
assert b[1] == 102
assert b[2] == 103

p = np.array([7, 8, 9])
q = p + 0

assert q[0] == 7
assert q[1] == 8
assert q[2] == 9

a_m1 = np.array([2, 3, 4])
b_m1 = a_m1 * 5

assert b_m1[0] == 10
assert b_m1[1] == 15
assert b_m1[2] == 20

p_m1 = np.array([7, 8, 9])
q_m1 = p_m1 * 0

assert q_m1[0] == 0
assert q_m1[1] == 0
assert q_m1[2] == 0
