import numpy as np

a = np.array([2, 3, 4])
b = a * 5

assert b[0] == 10
assert b[1] == 15
assert b[2] == 20

p = np.array([7, 8, 9])
q = p * 0

assert q[0] == 0
assert q[1] == 0
assert q[2] == 0
