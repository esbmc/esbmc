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
