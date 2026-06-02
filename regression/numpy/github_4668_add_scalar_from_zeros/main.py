import numpy as np

a = np.zeros(3)
b = a + 4

assert b[0] == 4
assert b[1] == 4
assert b[2] == 4
