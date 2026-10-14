import numpy as np

a = np.ones(3)
b = a + 5
assert b[0] == 6
assert b[1] == 6
assert b[2] == 6
