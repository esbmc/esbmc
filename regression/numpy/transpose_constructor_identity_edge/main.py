import numpy as np

# np.identity(n) and np.eye(n) both build an identity matrix but through
# distinct constructor calls; transpose of an identity matrix is itself.
a = np.identity(3)
b = np.transpose(a)
assert b[0][0] == 1
assert b[1][1] == 1
assert b[0][1] == 0

c = np.eye(3)
d = np.transpose(c)
assert d[0][0] == 1
assert d[1][1] == 1
assert d[0][1] == 0
