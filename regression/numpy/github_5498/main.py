# #5498: ndarray % scalar previously crashed the SMT backend (pointer modulo on
# the array). It is now modelled element-wise with Python/NumPy floored modulo
# (the result takes the sign of the divisor).
import numpy as np

a = np.array([4, 5, 6])
b = a % 3
assert b[0] == 1
assert b[1] == 2
assert b[2] == 0

c = np.array([-7, -1, 7])
d = c % 3
assert d[0] == 2
assert d[1] == 2
assert d[2] == 1
