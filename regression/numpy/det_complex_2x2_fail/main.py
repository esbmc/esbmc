import numpy as np
m = np.array([[1 + 1j, 2 + 0j], [3 + 0j, 4 + 0j]])
x = np.linalg.det(m)
assert x.real == -2.0
assert x.imag == 4.0
